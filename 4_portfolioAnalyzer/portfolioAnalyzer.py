"""
portfolioAnalyzer.py  —  v3.0
==============================
Portfolio-level growth & value stock analysis dashboard.

USAGE
-----
    python portfolioAnalyzer.py portfolio.csv [--backtest DAYS]

CSV FORMAT
----------
    ticker,shares[,cost_basis]
    NVDA,50,112.40
    AAPL,100,178.00
    MSFT,75
    CASH,10000

    cost_basis is optional (per-share average cost). Used to show
    unrealized P&L in the Overview and Rebalance tabs.
    The CASH row holds a dollar amount (not shares).

OUTPUT
------
    portfolioData/YYYY_MM_DD_portfolio.html  — opens in browser.

TABS
----
    Overview        Total value, P&L, portfolio beta, sector weights, signal counts
    Signals         BUY/HOLD/SELL with FV table, reason breakdown, context chips
    Technical       RSI, StochRSI, ATR, 52wk range, EMA, MACD, Bollinger
    Fundamentals    Revenue, margins, Ro40, dividend, short interest, earnings date
    Backtesting     Best valuation method per stock (MAPE + relative scoring)
    Rebalance       ATR trade sizing, concentration trims, earnings warnings
"""

import sys
import os
import csv
import json
import math
import datetime
import webbrowser
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuation_models import growth_adjusted_multiples, run_graham_number

# ── optional dependencies ──────────────────────────────────────────
try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False
    print("WARNING: yfinance not installed. pip install yfinance")

try:
    from tradingview_screener import Query, col
    _TV = True
except ImportError:
    _TV = False
    print("WARNING: tradingview-screener not installed. pip install tradingview-screener")

try:
    import pandas as pd
    import numpy as np
    _PANDAS = True
except ImportError:
    _PANDAS = False
    print("WARNING: pandas/numpy not installed. pip install pandas numpy")


# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────

RISK_FREE_RATE        = 0.043
EQUITY_RISK_PREMIUM   = 0.055
TERMINAL_GROWTH_RATE  = 0.030   # matches valuationMaster (was 0.025)
PROJECTION_YEARS      = 5       # DCF horizon
MARGIN_OF_SAFETY      = 0.20
MAX_POSITION_PCT      = 0.15   # 15% concentration trigger
TRIM_THRESHOLD_PCT    = 0.10   # 10% soft trim warning

# RSI thresholds
RSI_OVERSOLD      = 30
RSI_PULLBACK_LO   = 40
RSI_PULLBACK_HI   = 55
RSI_OVERBOUGHT    = 70

# Classification thresholds
GROWTH_REV_GROWTH_MIN  = 0.15   # 15% YoY revenue growth → growth stock
GROWTH_PE_MAX          = 40     # P/E > 40 likely growth / speculative
VALUE_PE_MAX           = 20     # P/E < 20 → value territory

# Backtest
DEFAULT_BACKTEST_DAYS  = 252    # 1 year


# ─────────────────────────────────────────────────────────────────
#  CSV READER
# ─────────────────────────────────────────────────────────────────

def load_portfolio(csv_path: str) -> dict:
    """
    Parse the portfolio CSV.
    Returns {"NVDA": {"shares":50,"cost_basis":112.40}, ...}
    Also accepts plain {"NVDA": 50} rows without cost_basis.
    CASH row value = dollar amount.
    """
    portfolio = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = [k.strip().lower() for k in row.keys()]
            vals = list(row.values())
            d = {k: v.strip() for k, v in zip(keys, vals)}
            ticker = d.get("ticker") or d.get("symbol") or d.get("stock")
            shares_str = d.get("shares") or d.get("quantity") or d.get("qty")
            if not ticker or not shares_str:
                continue
            ticker = ticker.upper().strip()
            try:
                shares_val = float(shares_str.replace(",", ""))
            except ValueError:
                continue
            cb_str = d.get("cost_basis") or d.get("cost") or d.get("avg_cost") or d.get("avg_price")
            cost_basis = None
            if cb_str:
                try: cost_basis = float(cb_str.replace(",", "").replace("$", ""))
                except ValueError: pass
            if ticker == "CASH":
                portfolio["CASH"] = {"shares": shares_val, "cost_basis": None}
            else:
                portfolio[ticker] = {"shares": shares_val, "cost_basis": cost_basis}
    return portfolio


# ─────────────────────────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────────────────────────

TV_FIELDS = [
    "name", "description", "close", "market_cap_basic",
    "total_shares_outstanding", "float_shares_outstanding",
    "total_debt", "free_cash_flow",
    "earnings_per_share_diluted_ttm", "earnings_per_share_basic_ttm",
    "gross_margin_percent_ttm", "net_income", "total_revenue",
    "operating_margin", "price_earnings_ttm",
    "enterprise_value_ebitda_ttm", "ebitda",
    "sector", "industry",
    "total_revenue_yoy_growth_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm",
    "beta_1_year",
    "cash_per_share_fy",
    # EPS forecasts — all horizons
    "earnings_per_share_forecast_fq",       # current quarter
    "earnings_per_share_forecast_next_fq",  # next quarter
    "earnings_per_share_forecast_next_fh",  # next half-year
    "earnings_per_share_forecast_next_fy",  # next full year (primary NTM)
    # Revenue forecasts — all horizons
    "revenue_forecast_fq",                  # current quarter
    "revenue_forecast_next_fq",             # next quarter
    "revenue_forecast_next_fh",             # next half-year
    "revenue_forecast_next_fy",             # next full year (primary NTM)
    # Analyst consensus intrinsic value
    # "fundamental_price",  # NOT a valid TV screener field
]

def fetch_tv_fundamentals(ticker: str) -> dict:
    """Fetch fundamentals from TradingView screener."""
    if not _TV:
        return {}
    try:
        _, df = (
            Query().select(*TV_FIELDS)
            .where(col("name") == ticker.upper())
            .limit(1).get_scanner_data()
        )
        if df.empty:
            return {}
        row = df.iloc[0]
        def safe(k, default=None):
            v = row.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return default
            return v

        price   = safe("close")
        mktcap  = safe("market_cap_basic", 0)
        debt    = safe("total_debt", 0) or 0
        cash_ps = safe("cash_per_share_fy", 0) or 0
        shares  = safe("total_shares_outstanding") or safe("float_shares_outstanding") or 1
        cash    = cash_ps * shares
        fcf     = safe("free_cash_flow")
        ni      = safe("net_income")
        rev     = safe("total_revenue", 0) or 0
        ebitda  = safe("ebitda")
        gm      = safe("gross_margin_percent_ttm")     # percent e.g. 74.5
        op_mar  = safe("operating_margin")
        beta    = safe("beta_1_year", 1.0) or 1.0
        eps     = safe("earnings_per_share_diluted_ttm") or safe("earnings_per_share_basic_ttm")
        # ── EPS forecasts — pull all horizons ─────────────────────
        eps_fq      = safe("earnings_per_share_forecast_fq")        # current qtr
        eps_next_fq = safe("earnings_per_share_forecast_next_fq")   # next qtr
        eps_next_fh = safe("earnings_per_share_forecast_next_fh")   # next half
        eps_next_fy = safe("earnings_per_share_forecast_next_fy")   # next full year

        # ── Revenue forecasts — pull all horizons ──────────────────
        rev_fq      = safe("revenue_forecast_fq")                   # current qtr
        rev_next_fq = safe("revenue_forecast_next_fq")              # next qtr
        rev_next_fh = safe("revenue_forecast_next_fh")              # next half
        rev_next_fy = safe("revenue_forecast_next_fy")              # next full year

        # ── Build best NTM EPS: hierarchy fy → fh×2 → (fq+nfq)×2 → fq×4 ──
        def best_ntm_eps():
            if eps_next_fy and eps_next_fy > 0:
                return round(eps_next_fy, 4), "fy"
            if eps_next_fh and eps_next_fh > 0:
                return round(eps_next_fh * 2, 4), "fh×2"
            if eps_fq and eps_next_fq and eps_fq > 0 and eps_next_fq > 0:
                return round((eps_fq + eps_next_fq) * 2, 4), "2q×2"
            if eps_next_fq and eps_next_fq > 0:
                return round(eps_next_fq * 4, 4), "nfq×4"
            if eps_fq and eps_fq > 0:
                return round(eps_fq * 4, 4), "fq×4"
            return None, None

        # ── Build best NTM Revenue: same hierarchy ──────────────────
        def best_ntm_rev():
            if rev_next_fy and rev_next_fy > 0:
                return round(rev_next_fy, 0), "fy"
            if rev_next_fh and rev_next_fh > 0:
                return round(rev_next_fh * 2, 0), "fh×2"
            if rev_fq and rev_next_fq and rev_fq > 0 and rev_next_fq > 0:
                return round((rev_fq + rev_next_fq) * 2, 0), "2q×2"
            if rev_next_fq and rev_next_fq > 0:
                return round(rev_next_fq * 4, 0), "nfq×4"
            if rev_fq and rev_fq > 0:
                return round(rev_fq * 4, 0), "fq×4"
            return None, None

        fwd_eps, fwd_eps_source = best_ntm_eps()
        fwd_rev, fwd_rev_source = best_ntm_rev()

        # TradingView returns growth as PERCENT (e.g. 22.4 = 22.4%)
        # We keep the raw percent AND a decimal version for different uses
        rev_growth_pct = safe("total_revenue_yoy_growth_ttm")   # percent, may be None
        eps_growth_pct = safe("earnings_per_share_diluted_yoy_growth_ttm")

        # EV components
        ev_approx = mktcap + debt - cash if mktcap else 0
        tv_ev_ebitda = safe("enterprise_value_ebitda_ttm")
        if not tv_ev_ebitda and ebitda and ebitda > 0 and ev_approx > 0:
            tv_ev_ebitda = ev_approx / ebitda

        # FCF margin and Rule-of-40 use the percent form
        fcf_margin = (fcf / rev * 100) if (fcf and rev > 0) else 0.0
        ro40 = (rev_growth_pct or 0) + fcf_margin

        # ── Growth rate derivation (mirrors valuationMaster priority order) ──
        # Start with FCF-margin proxy as fallback
        if   fcf_margin > 40: growth_proxy = 0.15
        elif fcf_margin > 20: growth_proxy = 0.12
        elif fcf_margin > 10: growth_proxy = 0.09
        else:                  growth_proxy = 0.06
        growth = growth_proxy
        growth_source = "FCF-margin proxy"

        # Priority 1: both fwd_eps AND rev_growth available → 50/50 blend
        if rev_growth_pct is not None and fwd_eps and eps and eps > 0 and fwd_eps != eps:
            eps_implied = (fwd_eps / eps) - 1.0
            rev_g_dec   = rev_growth_pct / 100.0
            growth      = max(0.02, min(eps_implied * 0.5 + rev_g_dec * 0.5, 0.80))
            growth_source = "analyst fwd EPS + TTM revenue (50/50 blend)"
        # Priority 2: fwd_eps only
        elif fwd_eps and eps and eps > 0 and fwd_eps > eps:
            eps_implied = (fwd_eps / eps) - 1.0
            growth      = max(0.02, min(eps_implied * 0.7 + growth_proxy * 0.3, 0.80))
            growth_source = "analyst fwd EPS (blended with proxy)"
        # Priority 3: TTM revenue growth
        elif rev_growth_pct is not None:
            growth = max(0.02, min(rev_growth_pct / 100.0, 0.80))
            growth_source = "TTM reported revenue growth"
        # Priority 4: implied from fwd_rev
        elif fwd_rev and fwd_rev > 0 and rev > 0:
            implied = (fwd_rev / rev) - 1.0
            if implied > 0:
                growth = max(0.02, min(implied, 0.80))
                growth_source = "implied from fwd/TTM revenue"

        fcf_per_share = (fcf / shares) if (fcf and shares and shares > 0) else None
        current_pe    = (price / eps)   if (price and eps and eps > 0)   else safe("price_earnings_ttm")
        current_pfcf  = (mktcap / fcf)  if (mktcap and fcf and fcf > 0) else None
        peg = (current_pe / rev_growth_pct) if (current_pe and rev_growth_pct and rev_growth_pct > 0) else None

        return {
            "ticker":           ticker.upper(),
            "company_name":     str(safe("description", ticker)).strip() or ticker,
            "price":            price,
            "market_cap":       mktcap,
            "sector":           str(safe("sector", "")) or None,
            "industry":         str(safe("industry", "")) or None,
            # Revenue
            "revenue":          rev,
            "fwd_rev":          fwd_rev,
            "fwd_rev_source":   fwd_rev_source,   # e.g. "fy" / "fh×2" / "2q×2"
            "rev_growth_pct":   rev_growth_pct,   # percent e.g. 22.4; may be None
            "rev_growth":       (rev_growth_pct / 100.0) if rev_growth_pct else 0.0,
            # Earnings
            "eps":              eps,
            "fwd_eps":          fwd_eps,
            "fwd_eps_source":   fwd_eps_source,   # e.g. "fy" / "fh×2" / "2q×2"
            # Raw per-horizon fields (for display / debugging)
            "eps_fq":           eps_fq,
            "eps_next_fq":      eps_next_fq,
            "eps_next_fh":      eps_next_fh,
            "eps_next_fy":      eps_next_fy,
            "rev_fq":           rev_fq,
            "rev_next_fq":      rev_next_fq,
            "rev_next_fh":      rev_next_fh,
            "rev_next_fy":      rev_next_fy,
            "eps_growth_pct":   eps_growth_pct,
            # TradingView analyst consensus intrinsic value (field not available in screener)
            "tv_fundamental_price": None,
            # Margins
            "gross_margin":     gm,               # percent e.g. 74.5
            "op_margin":        op_mar,
            "fcf_margin":       fcf_margin,        # percent
            "net_income":       ni,
            # Enterprise value  (keys must match valuationMaster exactly)
            "ebitda":           ebitda,
            "ebitda_method":    "TradingView TTM",
            "ev_approx":        ev_approx,
            "tv_ev_ebitda":     tv_ev_ebitda,
            # Capital structure
            "total_debt":       debt,
            "cash":             cash,
            "shares":           shares,
            "fcf":              fcf,
            "fcf_per_share":    fcf_per_share,
            # Market multiples
            "pe":               safe("price_earnings_ttm"),
            "current_pe":       current_pe,
            "current_pfcf":     current_pfcf,
            "peg":              peg,
            # Risk / growth
            "beta":             beta,
            "est_growth":       growth,            # used by ALL valuation methods
            "growth_source":    growth_source,
            "ro40":             ro40,
            # Derived ratios used by classify_stock + run_ev_ntm_revenue
            "ev_rev":           round(ev_approx / rev, 2) if (ev_approx and rev and rev > 0) else None,
            # WACC raw inputs (filled by fetch_wacc_inputs in main)
            "wacc_override":    None,
            "wacc_raw":         {},
        }
    except Exception as e:
        print("  TV fetch error for {}: {}".format(ticker, e))
        return {}


def fetch_price_history(ticker: str, days: int = 300) -> list:
    """
    Fetch daily OHLCV from yfinance.
    Returns list of {"date","open","high","low","close","volume"} oldest-first.
    """
    if not _YF:
        return []
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="{}d".format(days + 50))
        if hist.empty:
            return []
        result = []
        for dt, row in hist.iterrows():
            result.append({
                "date":   str(dt.date()),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return result[-days:]
    except Exception as e:
        print("  yfinance fetch error for {}: {}".format(ticker, e))
        return []


def fetch_wacc_inputs(ticker: str) -> dict:
    """Fetch yfinance TTM interest expense, effective tax rate, 5-yr beta, debt.
    Identical to valuationMaster.fetch_wacc_inputs()."""
    result = {"interest_expense": None, "income_tax_expense": None,
              "pretax_income": None, "total_debt_yf": None, "beta_yf": None}
    if not _YF:
        return result
    try:
        tk = yf.Ticker(ticker)
        def _first(df, names):
            if df is None or df.empty: return None
            for n in names:
                if n in df.index: return n
        def _ttm(df, names):
            k = _first(df, names)
            if k is None: return None
            vals = df.loc[k].dropna().iloc[:4]
            return float(vals.sum()) if len(vals) else None
        def _annual(df, names):
            k = _first(df, names)
            if k is None: return None
            row = df.loc[k].dropna()
            return float(row.iloc[0]) if len(row) else None

        IE  = ["Interest Expense","Interest Expense Non Operating","Net Interest Income","InterestExpense"]
        TAX = ["Tax Provision","Income Tax Expense","IncomeTaxExpense","Provision For Income Taxes"]
        PRE = ["Pretax Income","Income Before Tax","EarningsBeforeTax","PretaxIncome"]
        DBT = ["Total Debt","Long Term Debt And Capital Lease Obligation","Long Term Debt","TotalDebt"]

        q = tk.quarterly_income_stmt
        use_q = (q is not None and not q.empty and q.shape[1] >= 4)
        if use_q:
            ie = _ttm(q, IE); tax = _ttm(q, TAX); pre = _ttm(q, PRE)
        else:
            a = tk.income_stmt
            ie = _annual(a, IE); tax = _annual(a, TAX); pre = _annual(a, PRE)
        if ie is not None: ie = abs(ie)
        result.update({"interest_expense": ie, "income_tax_expense": tax,
                        "pretax_income": pre, "total_debt_yf": _annual(tk.balance_sheet, DBT)})
        try:
            b = tk.info.get("beta")
            if b is not None:
                b = float(b)
                result["beta_yf"] = b if 0.1 <= b <= 4.0 else None
        except Exception: pass
        # Book value per share — for Graham Number
        try:
            bvps = tk.info.get("bookValue")
            if bvps and float(bvps) > 0:
                result["book_value_ps"] = round(float(bvps), 2)
        except Exception: pass
    except Exception as e:
        print("  [WACC yf] {}: {}".format(ticker, e))
    return result


def fetch_analyst_forecasts(ticker: str) -> dict:
    """
    Fetch analyst consensus + extra market data from yfinance:
      rec_score        : +2 Strong Buy → -2 Strong Sell
      target_price     : consensus 12-mo price target
      surprise_avg_pct : avg EPS beat % last 4 quarters
      estimate_trend   : RAISING / LOWERING / STABLE
      week52_high/low  : 52-week range
      pct_from_52wk_high : % below 52-week high (momentum indicator)
      dividend_yield   : annual yield %
      short_pct_float  : short interest as % of float
      short_ratio      : days-to-cover
      earnings_date    : next earnings date string
    """
    out = {
        "target_price": None, "num_analysts": None, "recommendation": None,
        "rec_score": 0, "surprise_avg_pct": None,
        "estimate_trend": None, "estimate_detail": None,
        "week52_high": None, "week52_low": None, "pct_from_52wk_high": None,
        "dividend_yield": None,
        "short_pct_float": None, "short_ratio": None,
        "earnings_date": None,
    }
    if not _YF:
        return out
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try: info = tk.info or {}
        except Exception: pass

        out["target_price"]   = info.get("targetMeanPrice")
        out["num_analysts"]   = info.get("numberOfAnalystOpinions")

        # 52-week range
        h52 = info.get("fiftyTwoWeekHigh")
        l52 = info.get("fiftyTwoWeekLow")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        out["week52_high"] = h52
        out["week52_low"]  = l52
        if h52 and price:
            out["pct_from_52wk_high"] = round((price / h52 - 1) * 100, 1)

        # Dividend yield (as percent, e.g. 1.8)
        dy = info.get("dividendYield")
        if dy: out["dividend_yield"] = round(dy * 100, 2)

        # Short interest
        spf = info.get("shortPercentOfFloat")
        sr  = info.get("shortRatio")
        if spf: out["short_pct_float"] = round(spf * 100, 1)
        if sr:  out["short_ratio"]     = round(sr, 1)

        # Analyst recommendation
        rec_key = (info.get("recommendationKey") or "").lower()
        rec_map = {"strong_buy": ("Strong Buy", 2), "buy": ("Buy", 1),
                   "hold": ("Hold", 0), "underperform": ("Sell", -1), "sell": ("Strong Sell", -2)}
        if rec_key in rec_map:
            out["recommendation"], out["rec_score"] = rec_map[rec_key]

        # Next earnings date
        try:
            cal = tk.calendar
            if cal is not None:
                # yfinance returns calendar as dict or DataFrame depending on version
                if hasattr(cal, "get"):
                    ed = cal.get("Earnings Date")
                    if ed:
                        out["earnings_date"] = str(ed[0].date()) if hasattr(ed[0], "date") else str(ed[0])
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    ed = cal["Earnings Date"].iloc[0]
                    out["earnings_date"] = str(ed.date()) if hasattr(ed, "date") else str(ed)
        except Exception: pass

        # EPS beat/miss streak
        try:
            eh = tk.earnings_history
            if eh is not None and not eh.empty and "surprisePercent" in eh.columns:
                surprises = eh["surprisePercent"].dropna().tolist()[-4:]
                if surprises:
                    out["surprise_avg_pct"] = round(sum(surprises) / len(surprises) * 100, 1)
        except Exception: pass

        # Estimate revision trend from recent analyst actions
        try:
            recs = tk.recommendations
            if recs is not None and not recs.empty:
                cutoff = datetime.datetime.now() - datetime.timedelta(days=90)
                try: recent = recs[recs.index.tz_localize(None) >= cutoff]
                except Exception: recent = recs.tail(15)
                if not recent.empty:
                    grade_col = next((c for c in ["To Grade","toGrade","Action"] if c in recent.columns), None)
                    if grade_col:
                        grades = recent[grade_col].astype(str).str.lower().tolist()
                        ups   = sum(1 for g in grades if any(x in g for x in ["buy","outperform","overweight","upgrade"]))
                        downs = sum(1 for g in grades if any(x in g for x in ["sell","underperform","underweight","downgrade"]))
                        if ups > downs + 1:
                            out["estimate_trend"]  = "RAISING"
                            out["estimate_detail"] = "{} upgrades vs {} downgrades (90d)".format(ups, downs)
                        elif downs > ups + 1:
                            out["estimate_trend"]  = "LOWERING"
                            out["estimate_detail"] = "{} downgrades vs {} upgrades (90d)".format(downs, ups)
                        else:
                            out["estimate_trend"]  = "STABLE"
                            out["estimate_detail"] = "Mixed ({} up / {} down)".format(ups, downs)
        except Exception: pass
    except Exception as e:
        print("  [Analyst yf] {}: {}".format(ticker, e))
    return out


def fetch_market_sentiment(ticker: str) -> dict:
    """
    Fetch institutional/smart-money sentiment signals from yfinance:

    INSTITUTIONAL OWNERSHIP CHANGES
      inst_own_pct          : % of shares held by institutions (current)
      inst_own_change_pct   : quarter-over-quarter change in inst ownership %
      inst_buyers_count     : # institutions that increased/opened positions
      inst_sellers_count    : # institutions that decreased/closed positions
      inst_net_flow         : net BUY − SELL count (positive = net buying)

    SHORT INTEREST
      short_pct_float_change: Δ short % vs prior period (requires two readings)
      (short_pct_float is already in fetch_analyst_forecasts)

    OPTIONS SENTIMENT
      options_pc_ratio      : put/call ratio (>1.2 = bearish, <0.6 = bullish)
      (derived from open interest if available)

    INSIDER ACTIVITY
      insider_buy_count     : insider purchases last 6 months
      insider_sell_count    : insider sales last 6 months
      insider_net           : net insider flow direction

    ANALYST PRICE TARGET TREND
      target_high           : highest analyst price target
      target_low            : lowest analyst price target
      target_median         : median analyst price target
      target_trend          : RISING / FALLING / STABLE (vs 3mo ago)
    """
    out = {
        "inst_own_pct":           None,
        "inst_own_change_pct":    None,
        "inst_buyers_count":      None,
        "inst_sellers_count":     None,
        "inst_net_flow":          None,
        "inst_top_holders":       [],   # list of {name, pct, change}
        "insider_buy_count":      None,
        "insider_sell_count":     None,
        "insider_net":            None,
        "insider_transactions":   [],   # list of {name, role, type, shares, value}
        "target_high":            None,
        "target_low":             None,
        "target_median":          None,
        "target_trend":           None,
        "sentiment_score":        0,    # -3 to +3 composite
        "sentiment_label":        "Neutral",
        "sentiment_reasons":      [],
    }
    if not _YF:
        return out
    try:
        tk = yf.Ticker(ticker)

        # ── 1. Institutional ownership changes ────────────────────────
        try:
            inst = tk.institutional_holders
            if inst is not None and not inst.empty:
                # Current total inst ownership %
                if "% Out" in inst.columns:
                    pct_col = inst["% Out"].dropna()
                    if not pct_col.empty:
                        out["inst_own_pct"] = round(float(pct_col.sum()) * 100, 1)

                # Top 10 holders with change
                top_holders = []
                change_col = next((c for c in ["Change","pctChange","change"] if c in inst.columns), None)
                name_col   = next((c for c in ["Holder","holder","Name"] if c in inst.columns), None)
                pct_col_n  = next((c for c in ["% Out","pctHeld","Pct Held"] if c in inst.columns), None)
                if name_col and pct_col_n:
                    for _, row in inst.head(10).iterrows():
                        h = {
                            "name": str(row.get(name_col, "")),
                            "pct":  round(float(row.get(pct_col_n, 0) or 0) * 100, 2),
                            "change": None
                        }
                        if change_col:
                            chg = row.get(change_col)
                            if chg is not None and not (isinstance(chg, float) and math.isnan(chg)):
                                h["change"] = int(chg)
                        top_holders.append(h)
                out["inst_top_holders"] = top_holders
        except Exception: pass

        # ── 2. Major holders summary ──────────────────────────────────
        try:
            major = tk.major_holders
            if major is not None and not major.empty:
                # yfinance major_holders: first row is % held by insiders, second by institutions
                vals = major.iloc[:,0].tolist()
                if len(vals) >= 2:
                    # Try to extract inst ownership % from major holders summary
                    try:
                        inst_pct_str = str(vals[1]).replace("%","").strip()
                        inst_pct = float(inst_pct_str)
                        if out["inst_own_pct"] is None:
                            out["inst_own_pct"] = round(inst_pct, 1)
                    except Exception: pass
        except Exception: pass

        # ── 3. Institutional net flow (buyers vs sellers from most recent 13F) ──
        try:
            inst2 = tk.institutional_holders
            if inst2 is not None and not inst2.empty:
                change_col = next((c for c in ["Change","pctChange","change"] if c in inst2.columns), None)
                if change_col:
                    changes = inst2[change_col].dropna()
                    buyers  = int((changes > 0).sum())
                    sellers = int((changes < 0).sum())
                    out["inst_buyers_count"]  = buyers
                    out["inst_sellers_count"] = sellers
                    out["inst_net_flow"]      = buyers - sellers
        except Exception: pass

        # ── 4. Insider transactions ───────────────────────────────────
        try:
            insiders = tk.insider_transactions
            if insiders is not None and not insiders.empty:
                cutoff = datetime.datetime.now() - datetime.timedelta(days=180)
                # Filter to last 6 months
                try:
                    if "Start Date" in insiders.columns:
                        insiders = insiders[insiders["Start Date"] >= cutoff]
                    elif insiders.index.dtype == "datetime64[ns]":
                        insiders = insiders[insiders.index >= cutoff]
                except Exception: pass

                type_col    = next((c for c in ["Transaction","Insider Trading","Text","transactionText","Action","Type"] if c in insiders.columns), None)
                shares_col  = next((c for c in ["Shares","Shares Traded"] if c in insiders.columns), None)
                name_col_i  = next((c for c in ["Insider","Name"] if c in insiders.columns), None)
                role_col    = next((c for c in ["Relationship","Position","Title"] if c in insiders.columns), None)
                val_col     = next((c for c in ["Value","Transaction Value"] if c in insiders.columns), None)
                date_col    = next((c for c in ["Start Date","Date","Reported","startDate"] if c in insiders.columns), None)

                buys  = 0
                sells = 0
                txns  = []
                for idx, row in insiders.head(20).iterrows():
                    tx_type = str(row.get(type_col, "")).strip().lower() if type_col else ""
                    is_buy  = any(x in tx_type for x in ["purchase","buy","acquisition","exercis"])
                    is_sell = any(x in tx_type for x in ["sale","sell","dispos"])
                    if is_buy:  buys  += 1
                    if is_sell: sells += 1
                    if name_col_i:
                        sh = row.get(shares_col)
                        vl = row.get(val_col)
                        try:
                            sh = int(float(str(sh).replace(",","").replace("$","") or 0)) if sh else None
                        except Exception: sh = None
                        try:
                            vl = int(float(str(vl).replace(",","").replace("$","") or 0)) if vl else None
                        except Exception: vl = None
                        # Extract transaction date
                        tx_date = ""
                        try:
                            raw_dt = row.get(date_col) if date_col else None
                            if raw_dt is not None and str(raw_dt) not in ("", "NaT", "nan"):
                                tx_date = str(raw_dt)[:10]
                            elif hasattr(idx, "strftime"):
                                tx_date = idx.strftime("%Y-%m-%d")
                        except Exception: pass
                        # Human-readable type label
                        if is_buy:
                            tx_label = "BUY"
                        elif is_sell:
                            tx_label = "SELL"
                        elif tx_type:
                            tx_label = tx_type.title()[:18]
                        else:
                            tx_label = "OTHER"
                        txns.append({
                            "name":     str(row.get(name_col_i, "")),
                            "role":     str(row.get(role_col, "")) if role_col else "",
                            "type":     tx_label,
                            "is_buy":   is_buy,
                            "shares":   sh,
                            "value":    vl,
                            "date":     tx_date,
                        })
                out["insider_buy_count"]    = buys
                out["insider_sell_count"]   = sells
                out["insider_net"]          = buys - sells
                out["insider_transactions"] = txns[:10]
        except Exception: pass

        # ── 5. Analyst price target range ─────────────────────────────
        try:
            info = {}
            try: info = tk.info or {}
            except Exception: pass
            out["target_high"]   = info.get("targetHighPrice")
            out["target_low"]    = info.get("targetLowPrice")
            out["target_median"] = info.get("targetMedianPrice")
        except Exception: pass

        # ── 6. Composite sentiment score ─────────────────────────────
        score   = 0
        reasons = []

        inst_net = out.get("inst_net_flow")
        if inst_net is not None:
            if inst_net >= 5:
                score += 2; reasons.append("Institutional NET BUYING: {} more buyers than sellers".format(inst_net))
            elif inst_net >= 2:
                score += 1; reasons.append("Slight institutional net buying ({} net)".format(inst_net))
            elif inst_net <= -5:
                score -= 2; reasons.append("Institutional NET SELLING: {} more sellers than buyers".format(abs(inst_net)))
            elif inst_net <= -2:
                score -= 1; reasons.append("Slight institutional net selling ({} net)".format(abs(inst_net)))

        ins_net = out.get("insider_net")
        if ins_net is not None and (out.get("insider_buy_count",0) + out.get("insider_sell_count",0)) >= 2:
            if ins_net >= 2:
                score += 1; reasons.append("Insider NET BUYING: {} purchases vs {} sales (6mo)".format(
                    out["insider_buy_count"], out["insider_sell_count"]))
            elif ins_net <= -3:
                score -= 1; reasons.append("Heavy insider SELLING: {} sales vs {} buys (6mo)".format(
                    out["insider_sell_count"], out["insider_buy_count"]))

        inst_own = out.get("inst_own_pct")
        if inst_own and inst_own >= 70:
            score += 1; reasons.append("High institutional ownership {:.1f}% — strong smart-money conviction".format(inst_own))
        elif inst_own and inst_own < 20:
            score -= 1; reasons.append("Low institutional ownership {:.1f}% — limited smart-money interest".format(inst_own))

        if score >= 2:   label = "Bullish"
        elif score == 1: label = "Slightly Bullish"
        elif score <= -2: label = "Bearish"
        elif score == -1: label = "Slightly Bearish"
        else:            label = "Neutral"

        out["sentiment_score"]   = score
        out["sentiment_label"]   = label
        out["sentiment_reasons"] = reasons

    except Exception as e:
        print("  [Sentiment yf] {}: {}".format(ticker, e))
    return out


# ─────────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────

def compute_rsi(closes: list, period: int = 14) -> float:
    """Compute RSI for the most recent bar."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_ema(closes: list, period: int) -> list:
    """Compute EMA series."""
    if len(closes) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def compute_sma(closes: list, period: int) -> float:
    """Compute simple moving average of last `period` values."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def compute_macd(closes: list) -> dict:
    """Compute MACD (12,26,9). Returns {"macd","signal","hist"}."""
    if len(closes) < 35:
        return {}
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)
    # align lengths
    diff = len(ema12) - len(ema26)
    ema12 = ema12[diff:]
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    if len(macd_line) < 9:
        return {}
    signal = compute_ema(macd_line, 9)
    # align
    diff2 = len(macd_line) - len(signal)
    macd_aligned = macd_line[diff2:]
    hist = [m - s for m, s in zip(macd_aligned, signal)]
    return {
        "macd":   round(macd_aligned[-1], 4) if macd_aligned else None,
        "signal": round(signal[-1], 4) if signal else None,
        "hist":   round(hist[-1], 4) if hist else None,
        "macd_series":   macd_aligned[-60:],
        "signal_series": signal[-60:],
        "hist_series":   hist[-60:],
    }


def compute_bollinger(closes: list, period: int = 20, std_mult: float = 2.0) -> dict:
    """Compute Bollinger Bands. Returns {"upper","mid","lower","pct_b"}."""
    if len(closes) < period:
        return {}
    window = closes[-period:]
    mid  = sum(window) / period
    std  = (sum((x - mid)**2 for x in window) / period) ** 0.5
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return {
        "upper": round(upper, 2),
        "mid":   round(mid, 2),
        "lower": round(lower, 2),
        "pct_b": round(pct_b, 3),
    }


def detect_bb_touch_fail(bars: list) -> bool:
    """
    Bollinger Band touch-and-fail: price went outside upper band
    in the last 5 bars and has now closed back inside.
    """
    if len(bars) < 6:
        return False
    closes = [b["close"] for b in bars]
    bb_now = compute_bollinger(closes)
    if not bb_now:
        return False
    current_price = closes[-1]
    # Check if any of the last 5 days touched the upper band
    for i in range(-6, -1):
        sub_closes = closes[:i] if i != 0 else closes
        if len(sub_closes) < 20:
            continue
        bb_prev = compute_bollinger(sub_closes)
        if bb_prev and closes[i] >= bb_prev["upper"]:
            # Was outside upper band; now back inside
            if current_price < bb_now["upper"]:
                return True
    return False


def detect_macd_divergence(closes: list) -> bool:
    """
    Bearish divergence: price made a new high in last 20 bars
    but MACD histogram is lower than its previous peak.
    """
    if len(closes) < 40:
        return False
    macd_data = compute_macd(closes)
    if not macd_data or not macd_data.get("hist_series"):
        return False
    hist = macd_data["hist_series"]
    if len(hist) < 20:
        return False
    recent_hist  = hist[-20:]
    prev_hist    = hist[-40:-20] if len(hist) >= 40 else hist[:20]
    recent_close = closes[-20:]
    prev_close   = closes[-40:-20] if len(closes) >= 40 else closes[:20]
    price_made_new_high = max(recent_close) > max(prev_close)
    hist_lower = max(recent_hist) < max(prev_hist)
    return price_made_new_high and hist_lower


def compute_atr(bars: list, period: int = 14) -> float:
    """Average True Range — measures daily volatility for position sizing."""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        high  = bars[i]["high"]
        low   = bars[i]["low"]
        prev_close = bars[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    # Simple ATR (RMA/Wilder smoothing)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 4)


def compute_stoch_rsi(closes: list, rsi_period: int = 14, stoch_period: int = 14,
                      smooth_k: int = 3, smooth_d: int = 3) -> dict:
    """
    Stochastic RSI — oscillates 0-100.
    %K < 20 = oversold (buy zone), %K > 80 = overbought.
    Faster than plain RSI for catching intra-trend pullbacks.
    """
    if len(closes) < rsi_period + stoch_period + smooth_k + smooth_d + 5:
        return {}
    # Build RSI series
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_g = sum(gains[:rsi_period]) / rsi_period
    avg_l = sum(losses[:rsi_period]) / rsi_period
    rsi_series = []
    for i in range(rsi_period, len(gains)):
        avg_g = (avg_g * (rsi_period - 1) + gains[i]) / rsi_period
        avg_l = (avg_l * (rsi_period - 1) + losses[i]) / rsi_period
        rs = avg_g / avg_l if avg_l else 100
        rsi_series.append(100 - 100 / (1 + rs))
    if len(rsi_series) < stoch_period:
        return {}
    # Stochastic of RSI
    raw_k = []
    for i in range(stoch_period - 1, len(rsi_series)):
        window = rsi_series[i - stoch_period + 1: i + 1]
        lo, hi = min(window), max(window)
        raw_k.append((rsi_series[i] - lo) / (hi - lo) * 100 if (hi - lo) else 50)
    # Smooth %K
    def sma_s(lst, n):
        return [sum(lst[i:i+n])/n for i in range(len(lst)-n+1)]
    k_smooth = sma_s(raw_k, smooth_k)
    d_smooth = sma_s(k_smooth, smooth_d)
    if not k_smooth or not d_smooth:
        return {}
    return {
        "k": round(k_smooth[-1], 1),
        "d": round(d_smooth[-1], 1),
        "oversold":    k_smooth[-1] < 20,
        "overbought":  k_smooth[-1] > 80,
    }


def compute_technicals(bars: list) -> dict:
    """
    Compute all technical indicators for a stock.
    Returns dict with RSI, StochRSI, ATR, EMAs, SMAs, MACD, Bollinger, volume signals.
    """
    if len(bars) < 30:
        return {}
    closes  = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    rsi    = compute_rsi(closes)
    ema10  = compute_ema(closes, 10)
    ema50  = compute_ema(closes, 50)
    sma200 = compute_sma(closes, 200)
    macd   = compute_macd(closes)
    bb     = compute_bollinger(closes)

    price = closes[-1]

    ema10_val  = ema10[-1]  if ema10  else None
    ema50_val  = ema50[-1]  if ema50  else None

    # Volume signal: is today's volume > 1.5x 20-day avg?
    vol_avg20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else None
    vol_today = volumes[-1]
    high_volume = (vol_today > vol_avg20 * 1.5) if vol_avg20 else False

    # At EMA support (within 3%)?
    at_ema10  = (ema10_val  and abs(price - ema10_val)  / ema10_val  < 0.03) if ema10_val  else False
    at_ema50  = (ema50_val  and abs(price - ema50_val)  / ema50_val  < 0.03) if ema50_val  else False
    below_sma200 = (price < sma200 * 0.995) if sma200 else False

    bb_fail   = detect_bb_touch_fail(bars)
    macd_div  = detect_macd_divergence(closes)

    # ATR for position sizing
    atr = compute_atr(bars, 14)

    # Stochastic RSI for short-term timing
    stoch_rsi = compute_stoch_rsi(closes)

    # 52-week high/low from price history
    highs = [b["high"] for b in bars]
    lows  = [b["low"]  for b in bars]
    week52_high_bars = max(highs[-252:]) if len(highs) >= 20 else max(highs)
    week52_low_bars  = min(lows[-252:])  if len(lows)  >= 20 else min(lows)
    pct_from_52wk_high_bars = round((price / week52_high_bars - 1) * 100, 1) if week52_high_bars else None
    near_52wk_high = pct_from_52wk_high_bars is not None and pct_from_52wk_high_bars >= -5
    far_from_52wk_high = pct_from_52wk_high_bars is not None and pct_from_52wk_high_bars <= -30

    return {
        "rsi":               rsi,
        "stoch_rsi":         stoch_rsi,
        "atr":               atr,
        "ema10":             round(ema10_val, 2) if ema10_val else None,
        "ema50":             round(ema50_val, 2) if ema50_val else None,
        "sma200":            round(sma200, 2) if sma200 else None,
        "price":             round(price, 2),
        "bb":                bb,
        "macd":              macd,
        "high_volume":       high_volume,
        "at_ema10":          at_ema10,
        "at_ema50":          at_ema50,
        "below_sma200":      below_sma200,
        "bb_fail":           bb_fail,
        "macd_div":          macd_div,
        "pct_above_sma200":  round((price / sma200 - 1) * 100, 1) if sma200 else None,
        "week52_high":       round(week52_high_bars, 2),
        "week52_low":        round(week52_low_bars, 2),
        "pct_from_52wk_high": pct_from_52wk_high_bars,
        "near_52wk_high":    near_52wk_high,
        "far_from_52wk_high": far_from_52wk_high,
        "closes_60":         closes[-60:],
        "volumes_60":        volumes[-60:],
    }


# ─────────────────────────────────────────────────────────────────
#  STOCK CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

def classify_stock(fund: dict) -> dict:
    """
    Classify a stock as GROWTH, VALUE, or BLEND.
    Returns {"type", "confidence", "reasons"}.
    """
    reasons_g = []
    reasons_v = []
    score_g   = 0
    score_v   = 0

    rev_g = (fund.get("rev_growth") or 0)
    pe    = fund.get("pe")
    gm    = fund.get("gross_margin") or 0
    ro40  = fund.get("ro40") or 0
    ev_r  = fund.get("ev_rev")
    sector = (fund.get("sector") or "").lower()

    if rev_g >= 0.30:
        score_g += 3; reasons_g.append("Revenue growing >{:.0f}%".format(rev_g*100))
    elif rev_g >= 0.15:
        score_g += 2; reasons_g.append("Revenue growing {:.0f}%".format(rev_g*100))
    elif rev_g < 0.05:
        score_v += 2; reasons_v.append("Low revenue growth ({:.0f}%)".format(rev_g*100))

    if pe and pe > 40:
        score_g += 2; reasons_g.append("High P/E ({:.0f}x) — growth premium".format(pe))
    elif pe and pe < 18:
        score_v += 2; reasons_v.append("Low P/E ({:.0f}x) — value territory".format(pe))

    if gm >= 70:
        score_g += 2; reasons_g.append("High gross margin ({:.0f}%)".format(gm))
    elif gm < 35:
        score_v += 1; reasons_v.append("Low gross margin ({:.0f}%)".format(gm))

    if ro40 >= 40:
        score_g += 2; reasons_g.append("Rule-of-40 score {:.0f} (strong)".format(ro40))

    if ev_r and ev_r > 10:
        score_g += 1; reasons_g.append("High EV/Revenue ({:.1f}x)".format(ev_r))
    elif ev_r and ev_r < 3:
        score_v += 1; reasons_v.append("Low EV/Revenue ({:.1f}x)".format(ev_r))

    if any(s in sector for s in ["technology", "communication", "software"]):
        score_g += 1

    if score_g >= score_v + 2:
        stock_type = "GROWTH"
        confidence = min(100, int(score_g / (score_g + score_v + 1) * 100))
        reasons = reasons_g
    elif score_v >= score_g + 2:
        stock_type = "VALUE"
        confidence = min(100, int(score_v / (score_g + score_v + 1) * 100))
        reasons = reasons_v
    else:
        stock_type = "BLEND"
        confidence = 50
        reasons = reasons_g + reasons_v

    return {"type": stock_type, "confidence": confidence, "reasons": reasons}


# ─────────────────────────────────────────────────────────────────
#  VALUATION — identical logic to valuationMaster.py
# ─────────────────────────────────────────────────────────────────

def calculate_wacc(fund: dict) -> float:
    """
    Full WACC: (E/V)×Ke + (D/V)×Kd×(1−t)
    Uses yfinance TTM interest expense, effective tax rate, 5-yr beta.
    Identical to valuationMaster.calculate_wacc().
    """
    if fund.get("wacc_override"):
        return round(fund["wacc_override"], 4)

    raw    = fund.get("wacc_raw") or {}
    mktcap = fund.get("market_cap") or 0.0

    # Prefer yfinance 5-yr beta (standard) over TradingView 1-yr (noisy)
    beta_yf = raw.get("beta_yf")
    beta    = beta_yf if beta_yf is not None else (fund.get("beta") or 1.0)
    beta    = max(0.3, min(beta, 4.0))

    # Prefer yfinance balance-sheet debt
    debt_yf = raw.get("total_debt_yf")
    debt    = debt_yf if (debt_yf and debt_yf > 0) else (fund.get("total_debt") or 0.0)

    int_exp = raw.get("interest_expense")
    tax_exp = raw.get("income_tax_expense")
    pretax  = raw.get("pretax_income")

    ke = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

    # Effective tax rate — real TTM when available, 21% statutory fallback
    tax_rate = 0.21
    if pretax and pretax > 0 and tax_exp is not None:
        eff = max(0.0, tax_exp) / pretax
        if 0.0 <= eff <= 0.55:
            tax_rate = eff

    # Cost of debt — real TTM interest/debt when available, Rf+1.5% fallback
    kd = RISK_FREE_RATE + 0.015
    if int_exp and int_exp > 0 and debt > 0:
        kd_raw = int_exp / debt
        if 0.01 <= kd_raw <= 0.20:
            kd = kd_raw

    if mktcap > 0:
        V    = mktcap + debt
        wacc = (mktcap / V) * ke + (debt / V) * kd * (1.0 - tax_rate)
    else:
        wacc = ke

    return round(max(0.06, min(0.18, wacc)), 4)


def run_dcf(fund: dict) -> float:
    """
    2-stage 10-year FCF DCF:
      Stage 1 (years 1-5): high growth at est_growth, decaying each year
      Stage 2 (years 6-10): growth decays linearly from stage1_exit to terminal
      Terminal: Gordon growth model at TERMINAL_GROWTH_RATE

    This is far more realistic than a flat 5-year projection because:
    - Markets price in 10+ year cash flow streams for high-growth companies
    - Growth inevitably decays; modeling that decay avoids undervaluing
    - Stage 2 prevents the cliff from "all growth stops at year 5"
    """
    fcf    = fund.get("fcf")
    shares = fund.get("shares")
    nd     = (fund.get("total_debt") or 0) - (fund.get("cash") or 0)
    g      = fund.get("est_growth") or 0.05
    if not (fcf and fcf > 0 and shares and shares > 0):
        return None
    wacc = calculate_wacc(fund)

    # Stage 1: years 1-5, growth decays by 15% each year (compounding)
    DECAY_RATE = 0.15  # growth rate decays 15% per year
    g1 = g
    cf, pvs = fcf, []
    for yr in range(1, 6):
        g1 = g1 * (1 - DECAY_RATE)   # decay growth each year
        g1 = max(g1, TERMINAL_GROWTH_RATE * 2)  # floor at 2× terminal
        cf = cf * (1 + g1)
        pvs.append(cf / (1 + wacc) ** yr)
    stage1_exit_growth = g1

    # Stage 2: years 6-10, linear interpolation from stage1 exit to terminal
    for yr in range(6, 11):
        t = (yr - 5) / 5   # 0 at yr6, 1.0 at yr10
        g2 = stage1_exit_growth * (1 - t) + TERMINAL_GROWTH_RATE * t
        cf = cf * (1 + g2)
        pvs.append(cf / (1 + wacc) ** yr)

    # Terminal value at year 10
    term_val = cf * (1 + TERMINAL_GROWTH_RATE) / (wacc - TERMINAL_GROWTH_RATE)
    pv_term  = term_val / (1 + wacc) ** 10
    ev       = sum(pvs) + pv_term
    eq_val   = ev - nd
    if eq_val <= 0:
        return None
    return round(eq_val / shares, 2)


def run_pe_val(fund: dict, peer_pe: float = 22.0) -> float:
    """
    NTM P/E: uses FORWARD EPS (analyst consensus NTM) not TTM.
    TTM EPS is backward-looking; market prices on NTM earnings.
    3-way avg: (growth-target NTM P/E + industry-median + conserv) / 3
    """
    fwd_eps = fund.get("fwd_eps")
    eps_ttm = fund.get("eps")
    # Prefer NTM EPS — this is what the market actually prices on
    eps = fwd_eps if (fwd_eps and fwd_eps > 0) else eps_ttm
    if not (eps and eps > 0):
        return None
    mults      = growth_adjusted_multiples(fund.get("est_growth") or 0.05)
    fv_target  = eps * mults["target_pe"]
    fv_market  = eps * peer_pe
    fv_conserv = eps * mults["conserv_pe"]
    return round((fv_target + fv_market + fv_conserv) / 3, 2)


def run_pfcf_val(fund: dict, peer_pfcf: float = 22.0) -> float:
    """
    3-way avg: (growth-target P/FCF + industry-median + conserv) / 3
    Identical to valuationMaster.run_pfcf(). Replaces old flat (fcf/sh)×22.
    """
    fcf_ps = fund.get("fcf_per_share")
    if not (fcf_ps and fcf_ps > 0):
        fcf    = fund.get("fcf")
        shares = fund.get("shares")
        if not (fcf and fcf > 0 and shares and shares > 0):
            return None
        fcf_ps = fcf / shares
    mults      = growth_adjusted_multiples(fund.get("est_growth") or 0.05)
    fv_target  = fcf_ps * mults["target_pfcf"]
    fv_market  = fcf_ps * peer_pfcf
    fv_conserv = fcf_ps * mults["conserv_pfcf"]
    return round((fv_target + fv_market + fv_conserv) / 3, 2)


def run_ev_ebitda_val(fund: dict, peer_ev_ebitda: float = 14.0) -> float:
    """
    3-way avg: (growth-target EV/EBITDA + industry-median + conserv) / 3
    Identical to valuationMaster.run_ev_ebitda(). Replaces old flat ebitda×14.
    """
    ebitda = fund.get("ebitda")
    shares = fund.get("shares")
    nd     = (fund.get("total_debt") or 0) - (fund.get("cash") or 0)
    if not (ebitda and ebitda > 0 and shares and shares > 0):
        return None
    mults      = growth_adjusted_multiples(fund.get("est_growth") or 0.05)
    def ip(m): return (ebitda * m - nd) / shares
    fv_target  = ip(mults["target_eveb"])
    fv_market  = ip(peer_ev_ebitda)
    fv_conserv = ip(mults["conserv_eveb"])
    return round((fv_target + fv_market + fv_conserv) / 3, 2)


def run_forward_peg(fund: dict) -> float:
    """
    Fwd PEG — identical to valuationMaster.run_forward_peg().
    Uses analyst fwd_eps if available; falls back to TTM EPS × (1+growth).
    Returns base-case FV (PEG = 1.5×).
    """
    eps     = fund.get("eps") or 0
    fwd_eps = fund.get("fwd_eps")
    growth  = fund.get("est_growth") or 0
    price   = fund.get("price") or 0
    ntm_eps = fwd_eps if (fwd_eps and fwd_eps > 0) else (eps * (1 + growth) if eps > 0 else None)
    if not ntm_eps:
        return None
    g_pct = growth * 100
    if g_pct <= 0:
        return None
    # Cap at analyst target × 1.5 if available, else price × 8
    # Old cap of price×5 was artificially low for hypergrowth names
    target = an.get("target_price") if (an := fund.get("_analyst_cache")) else None
    cap = target * 1.5 if (target and target > 0) else (price * 8 if price else 99999)
    fv_b = max(0, min(ntm_eps * g_pct * 1.5, cap))
    return round(fv_b, 2) if fv_b > 0 else None


def run_ev_ntm_revenue(fund: dict) -> float:
    """
    EV/NTM Revenue with gross-margin tiers + growth bonus + 60/40 market blend.
    Identical to valuationMaster.run_ev_ntm_revenue().
    Replaces old flat 8× formula — low-margin stocks now get 1–5×, not 8×.
    """
    rev       = fund.get("revenue") or 0
    fwd_rev   = fund.get("fwd_rev")
    growth    = fund.get("est_growth") or 0
    gm        = fund.get("gross_margin")        # percent e.g. 74.5
    nd        = (fund.get("total_debt") or 0) - (fund.get("cash") or 0)
    shares    = fund.get("shares")
    ev_approx = fund.get("ev_approx") or 0
    if not (rev > 0 and shares and shares > 0):
        return None

    ntm_rev = fwd_rev if (fwd_rev and fwd_rev > 0) else rev * (1 + growth)

    # Step 1: gross-margin tier base ranges
    if   gm and gm >= 70: base_lo, base_mid, base_hi = 8.0, 15.0, 25.0
    elif gm and gm >= 50: base_lo, base_mid, base_hi = 4.0,  8.0, 16.0
    elif gm and gm >= 30: base_lo, base_mid, base_hi = 2.5,  4.5,  9.0
    else:                  base_lo, base_mid, base_hi = 1.0,  2.5,  5.0

    # Step 2: growth bonus above 15%
    g_pct = growth * 100
    if g_pct > 15:
        linear    = (g_pct - 15) / 10 * 1.5
        accel     = max(0, (g_pct - 40) / 10) ** 1.3
        bonus_mid = linear + accel
        bonus_lo, bonus_hi = bonus_mid * 0.5, bonus_mid * 1.4
    else:
        bonus_lo = bonus_mid = bonus_hi = 0.0

    tier_lo  = base_lo  + bonus_lo
    tier_mid = base_mid + bonus_mid
    tier_hi  = base_hi  + bonus_hi

    # Step 3: 60/40 blend with current market EV/Rev multiple
    cur = ev_approx / rev if rev > 0 else None
    if cur and cur > 1.0:
        lo  = tier_lo  * 0.6 + cur * 0.70 * 0.4
        mid = tier_mid * 0.6 + cur * 1.00 * 0.4
    else:
        lo, mid = tier_lo, tier_mid

    mid = max(2.0, round(mid, 1))
    def fv_at(m):
        eq = ntm_rev * m - nd
        return round(eq / shares, 2) if eq > 0 else None

    return fv_at(mid)


def fetch_market_benchmarks(sector: str = None, industry: str = None) -> dict:
    """
    Fetch live median P/E, P/FCF, EV/EBITDA for the stock's industry/sector.
    3-tier hierarchy: industry → sector → broad market.
    Identical to valuationMaster.fetch_market_benchmarks().
    """
    defaults = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0,
                "sector_name": "Broad Market", "peer_count": 0}
    if not _TV:
        return defaults

    BM_FIELDS = ["name", "market_cap_basic", "free_cash_flow",
                 "price_earnings_ttm", "enterprise_value_ebitda_ttm"]

    def _medians(df):
        if df.empty or len(df) < 10: return None
        bm = {}
        pe_col = df["price_earnings_ttm"].dropna()
        pe_col = pe_col[(pe_col > 5) & (pe_col < 100)]
        if len(pe_col) < 8: return None
        bm["pe"] = round(float(pe_col.median()), 1)
        df2 = df.copy()
        df2["pfcf"] = df2["market_cap_basic"] / df2["free_cash_flow"].replace(0, float("nan"))
        pf = df2["pfcf"].dropna(); pf = pf[(pf > 5) & (pf < 100)]
        bm["pfcf"] = round(float(pf.median()), 1) if len(pf) > 8 else bm["pe"]
        ev = df["enterprise_value_ebitda_ttm"].dropna(); ev = ev[(ev > 3) & (ev < 60)]
        bm["ev_ebitda"] = round(float(ev.median()), 1) if len(ev) > 8 else 14.0
        return bm

    for tier, filt, label in [
        ("industry", industry, industry),
        ("sector",   sector,   "{} (sector)".format(sector) if sector else None),
    ]:
        if not filt: continue
        try:
            cond = col("industry") == filt if tier == "industry" else col("sector") == filt
            _, df = (Query().select(*BM_FIELDS).where(
                cond, col("market_cap_basic") > 500e6, col("price_earnings_ttm") > 0)
                .order_by("market_cap_basic", ascending=False).limit(500).get_scanner_data())
            bm = _medians(df)
            if bm:
                bm["sector_name"] = label; bm["peer_count"] = len(df)
                return bm
        except Exception: pass

    try:
        _, df = (Query().select(*BM_FIELDS).where(
            col("market_cap_basic") > 5e9, col("price_earnings_ttm") > 0)
            .order_by("market_cap_basic", ascending=False).limit(200).get_scanner_data())
        bm = _medians(df)
        if bm:
            bm["sector_name"] = "Broad Market"; bm["peer_count"] = len(df)
            return bm
    except Exception: pass

    return defaults


def run_ntm_pe_val(fund: dict, benchmarks: dict = None) -> float:
    """
    NTM P/E method using analyst consensus NTM P/E multiple.
    This is how sell-side analysts actually set price targets:
      FV = NTM_EPS × analyst_consensus_NTM_PE

    If no consensus NTM P/E available, use growth-adjusted target P/E × NTM EPS.
    This is the most forward-looking and market-anchored method.
    """
    fwd_eps = fund.get("fwd_eps")
    if not (fwd_eps and fwd_eps > 0):
        return None

    # Try to get analyst consensus NTM P/E from benchmarks or estimate from growth
    bm = benchmarks or {}
    # Derive a reasonable NTM P/E: growth-adjusted but capped more generously
    growth = fund.get("est_growth") or 0.05
    g_pct  = growth * 100

    # PEG 1.5 on NTM EPS gives a market-anchored NTM P/E
    # For high-growth: NTM P/E = g_pct × 1.5, but no hard cap at 80
    # Large-cap tech trades at 25-50× NTM — use a sensible floor
    ntm_pe_growth = max(15.0, g_pct * 1.5)      # PEG 1.5 target
    ntm_pe_market = max(20.0, bm.get("pe", 22.0))  # industry peer
    ntm_pe_conserv = max(12.0, g_pct * 0.8)     # PEG 0.8 conservative

    fv = (fwd_eps * ntm_pe_growth + fwd_eps * ntm_pe_market + fwd_eps * ntm_pe_conserv) / 3
    return round(fv, 2)


def run_price_target_model(fund: dict, analyst: dict = None) -> float:
    """
    Analyst consensus 12-month price target model.
    The analyst target IS the market's 1-year fair value consensus — include it
    directly as a valuation method so it appears in the backtest comparison.

    Also returns a 3-way blended target:
      - Consensus mean target (from analyst)
      - Consensus median target
      - Implied from revenue/EPS growth extrapolation

    When analysts are close together (low dispersion), the signal is stronger.
    """
    an = analyst or {}
    mean_tgt   = an.get("target_price")
    median_tgt = an.get("target_median")
    high_tgt   = an.get("target_high")
    low_tgt    = an.get("target_low")

    if not mean_tgt:
        return None

    # Use mean as primary; blend with median if available
    if median_tgt and abs(mean_tgt - median_tgt) / mean_tgt < 0.20:
        blended = (mean_tgt * 0.6 + median_tgt * 0.4)
    else:
        blended = mean_tgt

    return round(blended, 2)


def run_tv_fundamental_price(fund: dict) -> float:
    """
    TradingView's own analyst consensus fundamental price.
    Directly represents what sell-side models are aggregating — use it
    as-is as a valuation anchor. Highest weight in backtest when MAPE is low.
    """
    tv_fv = fund.get("tv_fundamental_price")
    if tv_fv and tv_fv > 0:
        return round(float(tv_fv), 2)
    return None


def run_all_valuations(fund: dict, benchmarks: dict = None, analyst: dict = None) -> dict:
    """
    Run all 10 valuation methods.
    New in v3:
      - P/E now uses NTM EPS (not TTM)
      - DCF upgraded to 2-stage 10-year with growth decay
      - NTM P/E: market-anchored NTM earnings multiple
      - Price Target: analyst consensus 12-month target
      - Graham Number: for value stocks
      - TV Fundamental: TradingView's aggregated analyst consensus FV
    Returns {method_name: fair_value_float}.
    """
    bm = benchmarks or {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0}

    fvs = {}
    r = run_dcf(fund);                                  (fvs.__setitem__("DCF",         r) if r else None)
    r = run_pe_val(fund,        bm["pe"]);              (fvs.__setitem__("NTM P/E",     r) if r else None)
    r = run_pfcf_val(fund,      bm["pfcf"]);            (fvs.__setitem__("P/FCF",       r) if r else None)
    r = run_ev_ebitda_val(fund, bm["ev_ebitda"]);       (fvs.__setitem__("EV/EBITDA",   r) if r else None)
    r = run_forward_peg(fund);                          (fvs.__setitem__("Fwd PEG",     r) if r else None)
    r = run_ev_ntm_revenue(fund);                       (fvs.__setitem__("EV/NTM Rev",  r) if r else None)
    r = run_ntm_pe_val(fund, bm);                       (fvs.__setitem__("NTM P/E v2",  r) if r else None)
    r = run_price_target_model(fund, analyst);          (fvs.__setitem__("Analyst Tgt", r) if r else None)
    r = run_graham_number(fund);                        (fvs.__setitem__("Graham",      r) if r else None)
    r = run_tv_fundamental_price(fund);                 (fvs.__setitem__("TV Fund.",     r) if r else None)
    return fvs


# ─────────────────────────────────────────────────────────────────
#  12-MONTH PRICE PROJECTION
# ─────────────────────────────────────────────────────────────────

def project_price(fund: dict, tech: dict, analyst: dict,
                  benchmarks: dict = None) -> dict:
    """
    12-month forward price projection using 5 independent models.
    Returns a distribution (bear / base / bull) and weighted blended target.

    MODEL 1 — Analyst Consensus Extrapolation
      Uses the analyst 12-mo price target directly.
      Adjusted for estimate trend (raising/lowering upgrades/downgrades).

    MODEL 2 — NTM Earnings × Market Multiple
      Applies the forward P/E to analyst NTM EPS.
      forward_P/E is derived from growth-adjusted multiple, anchored to sector.
      This is how analysts build their own price targets.

    MODEL 3 — Revenue Momentum
      NTM revenue × current EV/Rev multiple, converted to per-share.
      Captures how revenue-driven stocks (AI, SaaS) are priced.

    MODEL 4 — Technical Price Projection
      52-week trend + EMA momentum extrapolation.
      Uses slope of 50-day EMA and RSI to project near-term momentum.

    MODEL 5 — DCF Justified Price
      The 2-stage DCF fair value. Anchors to fundamentals.

    WEIGHTING:
      Growth stocks:  Analyst 30%, NTM Earnings 25%, Revenue 20%, Technical 15%, DCF 10%
      Value stocks:   DCF 30%, NTM Earnings 25%, Analyst 20%, Technical 15%, Revenue 10%
      Blend:          Equal-weight 20% each
    """
    price   = fund.get("price") or 0
    if price <= 0:
        return {}

    an      = analyst or {}
    bm      = benchmarks or {"pe": 22.0}
    growth  = fund.get("est_growth") or 0.05
    fwd_eps = fund.get("fwd_eps")
    eps_ttm = fund.get("eps") or 0
    fwd_rev = fund.get("fwd_rev")
    revenue = fund.get("revenue") or 0
    shares  = fund.get("shares") or 1
    nd      = (fund.get("total_debt") or 0) - (fund.get("cash") or 0)
    ev_rev  = fund.get("ev_rev")
    gm      = fund.get("gross_margin") or 0
    stock_type = "GROWTH" if growth > 0.15 else "VALUE"

    projections = {}

    # ── Model 1: Analyst Consensus ────────────────────────────────
    analyst_target = an.get("target_price")
    if analyst_target and analyst_target > 0:
        # Adjust for estimate trend
        trend_adj = 0.0
        trend = an.get("estimate_trend")
        if trend == "RAISING":   trend_adj = 0.03   # +3% for upgrades
        elif trend == "LOWERING": trend_adj = -0.04  # -4% for downgrades
        projections["Analyst Consensus"] = {
            "base":  round(analyst_target * (1 + trend_adj), 2),
            "bull":  round(an.get("target_high", analyst_target * 1.15) or analyst_target * 1.15, 2),
            "bear":  round(an.get("target_low",  analyst_target * 0.80) or analyst_target * 0.80, 2),
            "note":  "Consensus ${:.0f} ({} analysts){}".format(
                analyst_target,
                an.get("num_analysts") or "?",
                ", estimates RAISING" if trend == "RAISING" else (", estimates LOWERING" if trend == "LOWERING" else ""))
        }

    # ── Model 2: NTM Earnings × Growth-Adjusted Forward P/E ──────
    ntm_eps = fwd_eps if (fwd_eps and fwd_eps > 0) else (eps_ttm * (1 + growth) if eps_ttm > 0 else None)
    if ntm_eps and ntm_eps > 0:
        g_pct = growth * 100
        # NTM P/E = PEG 1.5 × growth for growth stocks, industry median for value
        ntm_pe_base  = max(15.0, min(g_pct * 1.5, 120.0)) if stock_type == "GROWTH" else bm.get("pe", 18.0)
        ntm_pe_bull  = ntm_pe_base * 1.20   # multiple expansion scenario
        ntm_pe_bear  = ntm_pe_base * 0.75   # multiple compression scenario
        projections["NTM Earnings"] = {
            "base":  round(ntm_eps * ntm_pe_base, 2),
            "bull":  round(ntm_eps * ntm_pe_bull, 2),
            "bear":  round(ntm_eps * ntm_pe_bear, 2),
            "note":  "NTM EPS ${:.2f} × {:.0f}x P/E (PEG {:.1f})".format(
                ntm_eps, ntm_pe_base, ntm_pe_base / g_pct if g_pct > 0 else 0)
        }

    # ── Model 3: Revenue Momentum ─────────────────────────────────
    ntm_rev = fwd_rev if (fwd_rev and fwd_rev > 0) else (revenue * (1 + growth) if revenue > 0 else None)
    if ntm_rev and ntm_rev > 0 and shares > 0:
        # Use current EV/Rev as baseline; project forward
        cur_ev_rev = ev_rev or (fund.get("ev_approx", 0) / revenue if revenue > 0 else None)
        if cur_ev_rev and cur_ev_rev > 0:
            # Bear = 10% multiple compression, Bull = 10% expansion
            ev_base = ntm_rev * cur_ev_rev
            ev_bull = ntm_rev * cur_ev_rev * 1.10
            ev_bear = ntm_rev * cur_ev_rev * 0.85
            def to_price(ev): return max(0, round((ev - nd) / shares, 2))
            projections["Revenue Momentum"] = {
                "base": to_price(ev_base),
                "bull": to_price(ev_bull),
                "bear": to_price(ev_bear),
                "note": "NTM Rev ${:.0f}B × {:.1f}× EV/Rev".format(ntm_rev / 1e9, cur_ev_rev)
            }

    # ── Model 4: Technical Momentum Projection ────────────────────
    ema50    = tech.get("ema50")
    ema10    = tech.get("ema10")
    rsi      = tech.get("rsi")
    w52h     = tech.get("week52_high")
    w52l     = tech.get("week52_low")
    closes60 = tech.get("closes_60", [])
    if ema50 and price > 0 and len(closes60) >= 30:
        # 12-month momentum: extrapolate the 60-day trend annualized
        slope_pct = (closes60[-1] - closes60[0]) / closes60[0] / len(closes60) * 252
        # RSI adjustment: overbought dampens, oversold amplifies
        rsi_adj = 0
        if rsi:
            if rsi > 70: rsi_adj = -0.05
            elif rsi < 35: rsi_adj = 0.05
        base_return = slope_pct + rsi_adj
        # Cap at ±60% to avoid extrapolation runaway
        base_return = max(-0.50, min(0.60, base_return))
        projections["Technical Momentum"] = {
            "base": round(price * (1 + base_return), 2),
            "bull": round(price * (1 + base_return + 0.15), 2),
            "bear": round(price * (1 + base_return - 0.20), 2),
            "note": "60-day trend annualized {:.0f}% + RSI adj {:.0f}%".format(
                slope_pct * 100, rsi_adj * 100)
        }

    # ── Model 5: DCF Fair Value ───────────────────────────────────
    dcf_fv = run_dcf(fund)
    if dcf_fv and dcf_fv > 0:
        projections["DCF Fundamental"] = {
            "base": dcf_fv,
            "bull": round(dcf_fv * 1.25, 2),  # WACC -1% scenario
            "bear": round(dcf_fv * 0.75, 2),  # WACC +1% scenario
            "note": "2-stage 10yr DCF (WACC {:.1f}%)".format(calculate_wacc(fund) * 100)
        }

    # ── Model 6: TradingView Analyst Consensus FV ─────────────────
    tv_fv = fund.get("tv_fundamental_price")
    if tv_fv and tv_fv > 0 and price > 0:
        projections["TV Consensus FV"] = {
            "base": round(tv_fv, 2),
            "bull": round(tv_fv * 1.15, 2),
            "bear": round(tv_fv * 0.85, 2),
            "note": "TradingView aggregated analyst fundamental price"
        }

    if not projections:
        return {}

    # ── Weighted Blend ────────────────────────────────────────────
    if stock_type == "GROWTH":
        weights = {
            "Analyst Consensus":   0.30,
            "NTM Earnings":        0.25,
            "Revenue Momentum":    0.20,
            "Technical Momentum":  0.15,
            "DCF Fundamental":     0.10,
        }
    else:
        weights = {
            "DCF Fundamental":     0.30,
            "NTM Earnings":        0.25,
            "Analyst Consensus":   0.20,
            "Technical Momentum":  0.15,
            "Revenue Momentum":    0.10,
        }

    total_w, total_wv_base, total_wv_bull, total_wv_bear = 0, 0, 0, 0
    for name, proj in projections.items():
        w = weights.get(name, 0.15)
        total_w      += w
        total_wv_base += w * proj["base"]
        total_wv_bull += w * proj["bull"]
        total_wv_bear += w * proj["bear"]

    if total_w == 0:
        return {}

    blended_base = round(total_wv_base / total_w, 2)
    blended_bull = round(total_wv_bull / total_w, 2)
    blended_bear = round(total_wv_bear / total_w, 2)

    expected_return_pct = round((blended_base - price) / price * 100, 1)
    bull_return_pct     = round((blended_bull - price) / price * 100, 1)
    bear_return_pct     = round((blended_bear - price) / price * 100, 1)

    # Confidence: how tightly clustered are the models?
    bases = [p["base"] for p in projections.values() if p.get("base")]
    if len(bases) >= 2:
        mean_b = sum(bases) / len(bases)
        std_b  = (sum((b - mean_b)**2 for b in bases) / len(bases)) ** 0.5
        cv     = std_b / mean_b if mean_b > 0 else 1
        # CV <15% = High, 15-30% = Medium, >30% = Low
        if cv < 0.15:   confidence = "HIGH"
        elif cv < 0.30: confidence = "MEDIUM"
        else:           confidence = "LOW"
    else:
        confidence = "LOW"

    return {
        "models":               projections,
        "blended_base":         blended_base,
        "blended_bull":         blended_bull,
        "blended_bear":         blended_bear,
        "expected_return_pct":  expected_return_pct,
        "bull_return_pct":      bull_return_pct,
        "bear_return_pct":      bear_return_pct,
        "confidence":           confidence,
        "stock_type":           stock_type,
        "model_weights":        weights,
        "n_models":             len(projections),
    }


# ─────────────────────────────────────────────────────────────────
#  BACKTEST — find best method
# ─────────────────────────────────────────────────────────────────

def backtest_methods(fund: dict, bars: list, benchmarks: dict = None) -> dict:
    """
    MAPE-based backtest. benchmarks passed through to run_all_valuations
    so P/E, P/FCF, EV/EBITDA use live industry medians, not hardcoded defaults.
    Relative scoring: best method = 100, others proportional (avoids 0% floor).
    """
    if len(bars) < 60:
        return {}

    closes = [b["close"] for b in bars]
    fvs    = run_all_valuations(fund, benchmarks, analyst=None)
    mapes  = {}

    for method, fv in fvs.items():
        if fv and fv > 0:
            mape = sum(abs(fv - c) / c for c in closes) / len(closes) * 100
            mapes[method] = round(mape, 1)

    if not mapes:
        return {}

    best      = min(mapes, key=mapes.get)
    best_mape = mapes[best]

    # Relative scores: best = 100, others proportional
    # Add small epsilon so we never divide by zero
    rel = {m: round(min(100.0, best_mape / (e + 0.001) * 100), 1)
           for m, e in mapes.items()}

    return {
        "best_method":    best,
        "mape":           mapes,          # raw MAPE % per method
        "relative_score": rel,            # 0–100, best=100
        "fv":             fvs,            # fair value per method
        "best_fv":        fvs.get(best),
    }


# ─────────────────────────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_signal(fund: dict, tech: dict, classification: dict,
                    bt: dict, analyst: dict = None, benchmarks: dict = None) -> dict:
    """
    Composite BUY / HOLD / SELL signal with strength 0–100.
    Point sources:
      1. Valuation upside (best backtested method)   ±3
      2. Analyst consensus recommendation             ±2  [NEW]
      3. Analyst 12-month price target                ±2  [NEW]
      4. Estimate revision trend (upgrades/downgrades)±1  [NEW]
      5. EPS beat/miss streak                         ±1  [NEW]
      6. Forward revenue/EPS acceleration vs TTM      ±1  [NEW]
      7. Technical indicators (RSI, EMA, vol, SMA)    ±1–3
      8. Fundamental health (growth, Ro40, margin)    ±1–3
    """
    price      = fund.get("price") or 0
    stock_type = classification["type"]
    rev_g      = fund.get("rev_growth") or 0          # decimal 0.22 = 22%
    rev_g_pct  = fund.get("rev_growth_pct") or (rev_g * 100)
    gm         = fund.get("gross_margin") or 0
    ro40       = fund.get("ro40") or 0
    fwd_rev    = fund.get("fwd_rev")
    fwd_eps    = fund.get("fwd_eps")
    eps        = fund.get("eps") or 0
    an         = analyst or {}

    buy_pts  = 0
    sell_pts = 0
    hold_pts = 0
    reasons  = {"buy": [], "sell": [], "hold": []}

    # ── 1. Valuation — best backtested method ─────────────────────
    fvs = run_all_valuations(fund, benchmarks, analyst=an)
    best_method = bt.get("best_method") if bt else None
    if not best_method and fvs:
        best_method = list(fvs.keys())[0]
    best_fv = fvs.get(best_method) if best_method else None

    if best_fv and price > 0:
        upside = (best_fv - price) / price
        if upside > 0.20:
            buy_pts += 3
            reasons["buy"].append("{} FV ${:.0f} → {:.0f}% upside".format(
                best_method, best_fv, upside * 100))
        elif upside > 0.05:
            buy_pts += 1
            reasons["buy"].append("{} slight undervaluation ({:.0f}% upside)".format(
                best_method, upside * 100))
        elif upside < -0.20:
            sell_pts += 3
            reasons["sell"].append("{} FV ${:.0f} → {:.0f}% overvalued".format(
                best_method, best_fv, abs(upside) * 100))
        elif upside < -0.05:
            sell_pts += 1
            reasons["sell"].append("{} slight overvaluation ({:.0f}%)".format(
                best_method, abs(upside) * 100))
        else:
            hold_pts += 2
            reasons["hold"].append("{} fairly valued (±5%)".format(best_method))

    # ── 2. Analyst consensus recommendation ───────────────────────
    rec_score = an.get("rec_score", 0)
    rec_label = an.get("recommendation")
    n_analysts = an.get("num_analysts")
    cov = " ({} analysts)".format(n_analysts) if n_analysts else ""
    if rec_score >= 2:
        buy_pts += 2
        reasons["buy"].append("Analyst consensus: {}{}".format(rec_label, cov))
    elif rec_score == 1:
        buy_pts += 1
        reasons["buy"].append("Analyst consensus: {}{}".format(rec_label, cov))
    elif rec_score == -1:
        sell_pts += 1
        reasons["sell"].append("Analyst consensus: {}{}".format(rec_label, cov))
    elif rec_score <= -2:
        sell_pts += 2
        reasons["sell"].append("Analyst consensus: {}{}".format(rec_label, cov))

    # ── 3. Analyst 12-month price target ──────────────────────────
    target_price = an.get("target_price")
    if target_price and price > 0:
        t_upside = (target_price - price) / price
        if t_upside > 0.20:
            buy_pts += 2
            reasons["buy"].append("Analyst target ${:.0f} → {:.0f}% upside".format(
                target_price, t_upside * 100))
        elif t_upside > 0.05:
            buy_pts += 1
            reasons["buy"].append("Analyst target ${:.0f} ({:.0f}% upside)".format(
                target_price, t_upside * 100))
        elif t_upside < -0.15:
            sell_pts += 2
            reasons["sell"].append("Analyst target ${:.0f} below price ({:.0f}% downside)".format(
                target_price, abs(t_upside) * 100))
        elif t_upside < -0.05:
            sell_pts += 1
            reasons["sell"].append("Analyst target ${:.0f} slightly below price".format(target_price))

    # ── 4. Estimate revision trend ────────────────────────────────
    trend  = an.get("estimate_trend")
    detail = an.get("estimate_detail") or ""
    if trend == "RAISING":
        buy_pts += 1
        reasons["buy"].append("Estimates RAISING — {}".format(detail))
    elif trend == "LOWERING":
        sell_pts += 1
        reasons["sell"].append("Estimates LOWERING — {}".format(detail))

    # ── 5. EPS beat/miss streak ───────────────────────────────────
    surprise = an.get("surprise_avg_pct")
    if surprise is not None:
        if surprise > 5:
            buy_pts += 1
            reasons["buy"].append("Avg EPS beat +{:.1f}% last 4 quarters".format(surprise))
        elif surprise < -5:
            sell_pts += 1
            reasons["sell"].append("Avg EPS miss {:.1f}% last 4 quarters".format(surprise))

    # ── 6. Forward revenue/EPS acceleration vs TTM ────────────────
    if fwd_rev and fwd_rev > 0 and (fund.get("revenue") or 0) > 0:
        implied_fwd_g = (fwd_rev / fund["revenue"] - 1.0) * 100
        if implied_fwd_g > rev_g_pct + 3:
            buy_pts += 1
            reasons["buy"].append(
                "Revenue expected to ACCELERATE: fwd {:.0f}% vs TTM {:.0f}%".format(
                    implied_fwd_g, rev_g_pct))
        elif implied_fwd_g < rev_g_pct - 5:
            sell_pts += 1
            reasons["sell"].append(
                "Revenue expected to DECELERATE: fwd {:.0f}% vs TTM {:.0f}%".format(
                    implied_fwd_g, rev_g_pct))

    if fwd_eps and fwd_eps > 0 and eps > 0:
        eps_fwd_g = (fwd_eps / eps - 1.0) * 100
        if eps_fwd_g > 20:
            buy_pts += 1
            reasons["buy"].append(
                "EPS expected +{:.0f}% next year (${:.2f} → ${:.2f})".format(
                    eps_fwd_g, eps, fwd_eps))
        elif eps_fwd_g < -10:
            sell_pts += 1
            reasons["sell"].append(
                "EPS expected to fall {:.0f}% (${:.2f} → ${:.2f})".format(
                    abs(eps_fwd_g), eps, fwd_eps))

    # ── 7. Technical signals ──────────────────────────────────────
    rsi = tech.get("rsi")
    if rsi is not None:
        if stock_type == "GROWTH":
            if RSI_PULLBACK_LO <= rsi <= RSI_PULLBACK_HI:
                buy_pts += 2
                reasons["buy"].append("RSI {:.0f} — pullback buying zone (40–55)".format(rsi))
            elif rsi < RSI_OVERSOLD:
                buy_pts += 1
                reasons["buy"].append("RSI {:.0f} — oversold (check growth story)".format(rsi))
            elif rsi > RSI_OVERBOUGHT:
                sell_pts += 1
                reasons["sell"].append("RSI {:.0f} — overbought".format(rsi))
        else:
            if rsi < 35:
                buy_pts += 2
                reasons["buy"].append("RSI {:.0f} — oversold value opportunity".format(rsi))
            elif rsi > RSI_OVERBOUGHT:
                sell_pts += 1
                reasons["sell"].append("RSI {:.0f} — overbought".format(rsi))

    # Stochastic RSI — faster short-term timing signal
    srsi = tech.get("stoch_rsi", {})
    if srsi:
        if srsi.get("oversold"):
            buy_pts += 1
            reasons["buy"].append("Stoch RSI {:.0f} — short-term oversold (<20)".format(srsi["k"]))
        elif srsi.get("overbought"):
            sell_pts += 1
            reasons["sell"].append("Stoch RSI {:.0f} — short-term overbought (>80)".format(srsi["k"]))

    # 52-week high momentum
    pct52 = tech.get("pct_from_52wk_high")
    if pct52 is not None:
        if tech.get("near_52wk_high"):
            buy_pts += 1
            reasons["buy"].append("Within 5% of 52-week high — strong momentum")
        elif tech.get("far_from_52wk_high"):
            sell_pts += 1
            reasons["sell"].append("{:.0f}% below 52-week high — significant drawdown".format(abs(pct52)))

    if tech.get("at_ema50"):
        buy_pts += 2; reasons["buy"].append("Touching 50-day EMA support")
    if tech.get("at_ema10"):
        buy_pts += 1; reasons["buy"].append("Touching 10-day EMA pullback")
    if tech.get("high_volume"):
        buy_pts += 1; reasons["buy"].append("Volume >1.5× 20-day avg — institutional activity")
    if tech.get("below_sma200"):
        sell_pts += 3; reasons["sell"].append("Price BELOW 200-day SMA — major trend break")
    if tech.get("bb_fail"):
        sell_pts += 2; reasons["sell"].append("Bollinger Band touch-and-fail — exhaustion")
    if tech.get("macd_div"):
        sell_pts += 2; reasons["sell"].append("MACD bearish divergence — momentum fading")

    # ── 8. Fundamental health ─────────────────────────────────────
    if stock_type == "GROWTH":
        if rev_g >= 0.20:
            buy_pts += 1
            reasons["buy"].append("Revenue growing {:.0f}% — story intact".format(rev_g * 100))
        elif 0 < rev_g < 0.10:
            sell_pts += 2
            reasons["sell"].append("Revenue decelerating to {:.0f}% — story at risk".format(rev_g * 100))
        elif rev_g <= 0:
            sell_pts += 3
            reasons["sell"].append("Revenue declining — growth story broken")

        if ro40 >= 40:
            buy_pts += 1
            reasons["buy"].append("Rule-of-40: {:.0f} — excellent growth efficiency".format(ro40))
        elif ro40 < 20:
            sell_pts += 1
            reasons["sell"].append("Rule-of-40: {:.0f} — deteriorating".format(ro40))

        if gm >= 65:
            buy_pts += 1
            reasons["buy"].append("Gross margin {:.0f}% — strong pricing power".format(gm))

    # ── 9. Market sentiment — institutional + insider flow ──────────
    sentiment = an.get("_sentiment", {})
    sent_score = sentiment.get("sentiment_score", 0)
    if sent_score >= 2:
        buy_pts += 2
        for r in sentiment.get("sentiment_reasons", [])[:2]:
            reasons["buy"].append(r)
    elif sent_score == 1:
        buy_pts += 1
        for r in sentiment.get("sentiment_reasons", [])[:1]:
            reasons["buy"].append(r)
    elif sent_score <= -2:
        sell_pts += 2
        for r in sentiment.get("sentiment_reasons", [])[:2]:
            reasons["sell"].append(r)
    elif sent_score == -1:
        sell_pts += 1
        for r in sentiment.get("sentiment_reasons", [])[:1]:
            reasons["sell"].append(r)

    total = buy_pts + sell_pts + hold_pts + 1
    buy_score  = int(buy_pts  / total * 100)
    sell_score = int(sell_pts / total * 100)

    if buy_pts >= sell_pts + 3:
        signal = "BUY";            strength = min(100, buy_score + 20)
    elif sell_pts >= buy_pts + 3:
        signal = "SELL";           strength = min(100, sell_score + 20)
    elif buy_pts > sell_pts:
        signal = "HOLD / ACCUMULATE"; strength = min(75, buy_score + 10)
    elif sell_pts > buy_pts:
        signal = "HOLD / TRIM";    strength = min(75, sell_score + 10)
    else:
        signal = "HOLD";           strength = 50

    return {
        "signal":      signal,
        "strength":    strength,
        "buy_pts":     buy_pts,
        "sell_pts":    sell_pts,
        "reasons":     reasons,
        "best_method": best_method,
        "best_fv":     best_fv,
        "all_fvs":     fvs,
        "analyst":     an,
    }


# ─────────────────────────────────────────────────────────────────
#  PORTFOLIO METRICS
# ─────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(stocks: dict, cash: float) -> dict:
    """
    Computes portfolio-level metrics including:
    - Allocations, concentration warnings
    - Weighted-avg fundamentals (beta, growth, margin, upside)
    - Sector/industry weights
    - Per-position P&L (when cost_basis provided)
    - Portfolio correlation clustering warning (using price-return overlap)
    """
    total_equity = sum(s["value"] for s in stocks.values())
    total_value  = total_equity + cash

    allocations = {}
    total_pnl   = 0.0
    total_cost   = 0.0
    for ticker, s in stocks.items():
        val    = s["value"]
        shares = s["shares"]
        fund   = s.get("fund", {})
        price  = fund.get("price") or 0
        cb     = s.get("cost_basis")    # per-share cost basis
        pnl    = None
        pnl_pct = None
        cost_total = None
        if cb and cb > 0 and shares > 0 and price > 0:
            cost_total = cb * shares
            pnl        = val - cost_total
            pnl_pct    = (price / cb - 1) * 100
            total_pnl  += pnl
            total_cost += cost_total
        allocations[ticker] = {
            "value":      val,
            "pct":        val / total_value * 100 if total_value > 0 else 0,
            "shares":     shares,
            "cost_basis": cb,
            "pnl":        pnl,
            "pnl_pct":    pnl_pct,
            "cost_total": cost_total,
        }
    allocations["CASH"] = {
        "value": cash, "pct": cash / total_value * 100 if total_value > 0 else 0,
        "shares": 0, "cost_basis": None, "pnl": None, "pnl_pct": None, "cost_total": None,
    }

    # Concentration warnings
    concentration_warns = [
        t for t, a in allocations.items()
        if t != "CASH" and a["pct"] > MAX_POSITION_PCT * 100
    ]
    trim_warns = [
        t for t, a in allocations.items()
        if t != "CASH" and a["pct"] > TRIM_THRESHOLD_PCT * 100 and t not in concentration_warns
    ]

    # ── Weighted-average portfolio fundamentals ──
    def wavg(field, source="fund", default=None):
        """Weighted average of a fund field, weighted by equity value."""
        total_w, total_wv = 0.0, 0.0
        for ticker, s in stocks.items():
            w   = s["value"]
            obj = s.get(source, {}) if source != "fund" else s.get("fund", {})
            v   = (obj or {}).get(field)
            if v is not None:
                total_w  += w
                total_wv += w * v
        return round(total_wv / total_w, 3) if total_w > 0 else default

    wavg_beta    = wavg("beta")
    wavg_growth  = wavg("est_growth")
    wavg_gm      = wavg("gross_margin")
    wavg_ro40    = wavg("ro40")

    # Weighted-avg upside (using best_fv from each stock's signal)
    upside_w, upside_wv = 0.0, 0.0
    for ticker, s in stocks.items():
        sig   = s.get("signal", {})
        fund  = s.get("fund", {})
        price = fund.get("price") or 0
        fv    = sig.get("best_fv")
        w     = s["value"]
        if fv and price > 0:
            upside_w  += w
            upside_wv += w * (fv / price - 1) * 100
    wavg_upside = round(upside_wv / upside_w, 1) if upside_w > 0 else None

    # ── Sector weights ──
    sector_weights = {}
    for ticker, s in stocks.items():
        sector = (s.get("fund", {}) or {}).get("sector") or "Unknown"
        w = allocations[ticker]["pct"]
        sector_weights[sector] = sector_weights.get(sector, 0) + w

    # ── Simple correlation clustering ──
    # Flag if any 3+ tickers share the same sector AND combined weight > 40%
    sector_tickers = {}
    for ticker, s in stocks.items():
        sector = (s.get("fund", {}) or {}).get("sector") or "Unknown"
        sector_tickers.setdefault(sector, []).append(ticker)
    concentration_sectors = {
        sec: tickers for sec, tickers in sector_tickers.items()
        if len(tickers) >= 3 and sector_weights.get(sec, 0) > 40
    }

    return {
        "total_value":            total_value,
        "total_equity":           total_equity,
        "cash":                   cash,
        "cash_pct":               cash / total_value * 100 if total_value > 0 else 0,
        "allocations":            allocations,
        "concentration_warns":    concentration_warns,
        "trim_warns":             trim_warns,
        "n_stocks":               len(stocks),
        # P&L
        "total_pnl":              total_pnl if total_cost > 0 else None,
        "total_pnl_pct":          (total_pnl / total_cost * 100) if total_cost > 0 else None,
        "has_cost_basis":         total_cost > 0,
        # Weighted portfolio fundamentals
        "wavg_beta":              wavg_beta,
        "wavg_growth":            wavg_growth,
        "wavg_gross_margin":      wavg_gm,
        "wavg_ro40":              wavg_ro40,
        "wavg_upside":            wavg_upside,
        # Sector breakdown
        "sector_weights":         sector_weights,
        "concentration_sectors":  concentration_sectors,
    }


# ─────────────────────────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────────────────────────

def fmt_money(v: float, decimals: int = 2) -> str:
    if v is None: return "—"
    if abs(v) >= 1e9:  return "${:.2f}B".format(v / 1e9)
    if abs(v) >= 1e6:  return "${:.1f}M".format(v / 1e6)
    if abs(v) >= 1000: return ("${:,." + str(decimals) + "f}").format(v)
    return ("${:." + str(decimals) + "f}").format(v)

def fmt_pct(v: float, signed: bool = False) -> str:
    if v is None: return "—"
    s = "+" if (signed and v >= 0) else ""
    return "{}{}%".format(s, round(v, 1))

def fmt_num(v: float, dp: int = 1) -> str:
    if v is None: return "—"
    return "{:.{}f}".format(dp, v)

def signal_color(signal: str) -> str:
    if "BUY" in signal:   return "#00c896"
    if "SELL" in signal:  return "#e05c5c"
    return "#f0a500"

def signal_bg(signal: str) -> str:
    if "BUY" in signal:   return "rgba(0,200,150,0.12)"
    if "SELL" in signal:  return "rgba(224,92,92,0.12)"
    return "rgba(240,165,0,0.12)"

def type_color(t: str) -> str:
    if t == "GROWTH": return "#c45aff"
    if t == "VALUE":  return "#4f8ef7"
    return "#f0a500"

def rsi_color(rsi: float) -> str:
    if rsi is None: return "#6b7194"
    if rsi < 35:    return "#00c896"
    if rsi > 70:    return "#e05c5c"
    if 40 <= rsi <= 55: return "#00c896"
    return "#f0a500"

def mini_sparkline(values: list, color: str = "#4f8ef7", w: int = 120, h: int = 40) -> str:
    """Generate an inline SVG sparkline."""
    if not values or len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * w
        y = h - (v - mn) / rng * h
        pts.append("{:.1f},{:.1f}".format(x, y))
    path = "M " + " L ".join(pts)
    return (
        '<svg width="{}" height="{}" viewBox="0 0 {} {}" style="display:block;">'
        '<path d="{}" stroke="{}" stroke-width="1.5" fill="none" opacity="0.85"/>'
        '</svg>'
    ).format(w, h, w, h, path, color)


def atr_position_size(total_portfolio: float, atr: float, price: float,
                      risk_pct: float = 0.01) -> dict:
    """
    ATR-based position sizing (1% portfolio risk per trade, 2-ATR stop).
    Shares = (portfolio × risk_pct) / (2 × ATR)
    """
    if not (atr and atr > 0 and price and price > 0 and total_portfolio > 0):
        return {}
    risk_budget    = total_portfolio * risk_pct
    stop_distance  = 2.0 * atr
    shares         = max(1, int(risk_budget / stop_distance))
    dollar_amount  = shares * price
    stop_price     = price - stop_distance
    return {
        "shares":        shares,
        "dollar_amount": round(dollar_amount, 2),
        "stop_price":    round(stop_price, 2),
        "risk_budget":   round(risk_budget, 2),
        "stop_distance": round(stop_distance, 2),
    }


def trim_to_target(current_shares: float, current_price: float,
                   current_pct: float, target_pct: float,
                   total_portfolio: float) -> dict:
    """
    Exact shares/dollars needed to reach target_pct from current_pct.
    """
    if not (current_price > 0 and total_portfolio > 0):
        return {}
    target_value  = total_portfolio * target_pct / 100
    current_value = current_shares * current_price
    delta_value   = target_value - current_value
    delta_shares  = max(1, int(abs(delta_value) / current_price))
    action        = "ADD" if delta_value > 0 else "TRIM"
    return {
        "action":       action,
        "shares":       delta_shares,
        "dollars":      round(abs(delta_value), 2),
        "target_value": round(target_value, 2),
    }


# ─────────────────────────────────────────────────────────────────
#  HTML REPORT BUILDER
# ─────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg:       #0d0f14;
  --surface:  #14171f;
  --surface2: #1c2030;
  --border:   #252a3a;
  --text:     #e8eaf2;
  --muted:    #6b7194;
  --accent:   #4f8ef7;
  --up:       #00c896;
  --down:     #e05c5c;
  --warn:     #f0a500;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Syne', sans-serif;
  min-height: 100vh;
  padding: 0 0 80px;
}
.hero {
  background: linear-gradient(135deg, #0d0f14 0%, #0f1520 50%, #0d0f14 100%);
  border-bottom: 1px solid var(--border);
  padding: 56px 48px 44px;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -120px; right: -120px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(79,142,247,0.07) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-label { font-family:'DM Mono',monospace; font-size:11px; letter-spacing:3px; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }
.hero-title { font-family:'DM Serif Display',serif; font-size:clamp(32px,4vw,54px); line-height:1.05; color:var(--text); margin-bottom:8px; }
.hero-subtitle { font-size:14px; color:var(--muted); margin-bottom:32px; }
.hero-meta { display:flex; gap:40px; flex-wrap:wrap; }
.meta-item { display:flex; flex-direction:column; gap:4px; }
.meta-label { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); }
.meta-value { font-size:22px; font-weight:700; }
.meta-value.green { color:var(--up); }
.meta-value.red   { color:var(--down); }
.meta-value.warn  { color:var(--warn); }

.tab-bar { display:flex; border-bottom:2px solid var(--border); background:var(--bg); position:sticky; top:0; z-index:20; overflow-x:auto; }
.tab-btn { background:none; border:none; border-bottom:3px solid transparent; margin-bottom:-2px; padding:16px 28px; color:var(--muted); font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2px; text-transform:uppercase; cursor:pointer; transition:color .15s,border-color .15s; white-space:nowrap; }
.tab-btn:hover { color:var(--text); }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

.container { max-width:1280px; margin:0 auto; padding:0 48px; }
section { margin-top:52px; }
.section-label { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:3px; text-transform:uppercase; color:var(--muted); margin-bottom:20px; padding-bottom:12px; border-bottom:1px solid var(--border); }

/* Portfolio table */
.port-table { width:100%; border-collapse:collapse; font-size:13px; }
.port-table th { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); padding:10px 14px; text-align:left; border-bottom:2px solid var(--border); }
.port-table th:not(:first-child):not(:nth-child(2)) { text-align:right; }
.port-table td { padding:12px 14px; border-bottom:1px solid var(--border); vertical-align:middle; }
.port-table td:not(:first-child):not(:nth-child(2)) { text-align:right; font-family:'DM Mono',monospace; }
.port-table tr:hover td { background:var(--surface2); }
.port-table tr:last-child td { border-bottom:none; }

/* Signal pill */
.signal-pill { display:inline-block; padding:5px 14px; border-radius:20px; font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase; font-weight:700; }

/* Type badge */
.type-badge { display:inline-block; padding:2px 10px; border-radius:4px; font-family:'DM Mono',monospace; font-size:9px; letter-spacing:1px; text-transform:uppercase; font-weight:700; border:1px solid; }

/* Allocation bar */
.alloc-bar { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.alloc-name { font-family:'DM Mono',monospace; font-size:12px; width:70px; flex-shrink:0; }
.alloc-track { flex:1; background:var(--surface2); border-radius:3px; height:20px; position:relative; overflow:hidden; }
.alloc-fill { height:100%; border-radius:3px; min-width:2px; }
.alloc-val { font-family:'DM Mono',monospace; font-size:11px; color:var(--muted); white-space:nowrap; width:80px; text-align:right; }

/* Stat card */
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:20px 22px; }
.stat-label { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.stat-value { font-family:'DM Serif Display',serif; font-size:32px; color:var(--text); line-height:1; }
.stat-sub { font-size:12px; color:var(--muted); margin-top:6px; }

/* Stock detail card */
.stock-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:20px; overflow:hidden; }
.stock-card-header { display:flex; align-items:center; gap:16px; padding:20px 24px 16px; border-bottom:1px solid var(--border); flex-wrap:wrap; }
.stock-card-body { padding:20px 24px; }
.sc-ticker { font-family:'DM Serif Display',serif; font-size:26px; color:var(--accent); }
.sc-name { font-size:13px; color:var(--muted); margin-top:2px; }
.sc-price { margin-left:auto; text-align:right; }
.sc-price-val { font-family:'DM Mono',monospace; font-size:20px; font-weight:700; }
.sc-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }
.sc-metric { display:flex; flex-direction:column; gap:3px; }
.sc-metric-label { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); }
.sc-metric-value { font-family:'DM Mono',monospace; font-size:14px; font-weight:600; }

/* Technical indicator row */
.tech-row { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); font-size:13px; }
.tech-row:last-child { border-bottom:none; }
.tech-indicator { width:140px; color:var(--muted); font-family:'DM Mono',monospace; font-size:11px; }
.tech-value { font-family:'DM Mono',monospace; font-size:13px; font-weight:600; width:80px; }
.tech-signal { font-size:12px; color:var(--muted); flex:1; }
.tech-icon { font-size:16px; }

/* Signal reason list */
.reason-list { list-style:none; padding:0; margin:0; }
.reason-list li { font-size:12px; color:var(--muted); padding:4px 0; line-height:1.5; }
.reason-list li::before { content:"→ "; color:var(--accent); }
.reason-list.sell li::before { color:var(--down); }
.reason-list.hold li::before { color:var(--warn); }

/* Rebalance card */
.rebal-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:22px 24px; margin-bottom:16px; display:flex; align-items:flex-start; gap:20px; }
.rebal-action { font-family:'DM Serif Display',serif; font-size:40px; line-height:1; }
.rebal-body { flex:1; }
.rebal-ticker { font-weight:700; font-size:16px; margin-bottom:4px; }
.rebal-reason { font-size:13px; color:var(--muted); line-height:1.6; }

/* Fund table */
.fund-block { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:16px; }
.fund-block-title { padding:12px 18px; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; background:var(--surface2); border-bottom:1px solid var(--border); color:var(--muted); }
.fund-row { display:flex; justify-content:space-between; align-items:center; padding:10px 18px; border-bottom:1px solid var(--border); font-size:13px; }
.fund-row:last-child { border-bottom:none; }
.fund-key { color:var(--muted); }
.fund-val { font-family:'DM Mono',monospace; font-weight:500; }

/* Progress bar for strength */
.strength-bar { height:6px; border-radius:3px; background:var(--surface2); margin-top:6px; overflow:hidden; }
.strength-fill { height:100%; border-radius:3px; }

/* Warning banner */
.warn-banner { background:rgba(240,165,0,0.08); border:1px solid rgba(240,165,0,0.25); border-left:4px solid var(--warn); border-radius:8px; padding:16px 20px; margin-bottom:16px; font-size:13px; color:var(--muted); }
.warn-banner strong { color:var(--warn); }

.danger-banner { background:rgba(224,92,92,0.08); border:1px solid rgba(224,92,92,0.25); border-left:4px solid var(--down); border-radius:8px; padding:16px 20px; margin-bottom:16px; font-size:13px; color:var(--muted); }
.danger-banner strong { color:var(--down); }

.footer { margin-top:72px; padding:32px 48px; border-top:1px solid var(--border); text-align:center; font-size:12px; color:var(--muted); line-height:1.8; }

@media (max-width:700px) {
  .hero { padding:36px 24px 32px; }
  .container { padding:0 20px; }
  .tab-btn { padding:14px 18px; }
}
"""

JS = """
function switchTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
}
"""


def _build_risk_tab_html(stocks_data: dict, metrics: dict) -> str:
    """
    Portfolio Risk tab:
      • Portfolio VaR — historical simulation (95%/99%, 1-day & 10-day)
      • Correlation matrix — SVG heatmap of pairwise Pearson correlations
      • Per-stock stats — annualised vol, Sharpe (Rf = 4.3%), max drawdown
    All maths are pure Python — no external dependencies beyond stdlib.
    """
    RF_ANNUAL = 0.043
    MIN_OBS   = 30   # minimum common bar count to include a ticker

    # ── 1. Build daily-return series per ticker ────────────────────────────
    tickers_all = [t for t in stocks_data if stocks_data[t].get("bars")]

    rets_map = {}   # {ticker: {date_str: daily_return}}
    for ticker in tickers_all:
        bars = stocks_data[ticker]["bars"]
        d = {}
        for i in range(1, len(bars)):
            p0 = bars[i - 1]["close"]
            p1 = bars[i]["close"]
            if p0 and p0 > 0:
                d[bars[i]["date"]] = (p1 - p0) / p0
        if len(d) >= MIN_OBS:
            rets_map[ticker] = d

    tickers = sorted(rets_map.keys())

    if len(tickers) < 2:
        return (
            '<div id="tab-risk" class="tab-panel">'
            '<div class="container" style="padding-top:48px;">'
            '<div class="section-label">Portfolio Risk</div>'
            '<p style="color:var(--muted);font-size:14px;">'
            'Insufficient price history to compute risk metrics. '
            'At least 2 holdings with 30+ days of data are required.'
            '</p></div></div>'
        )

    # ── 2. Align to common date intersection ──────────────────────────────
    date_sets    = [set(rets_map[t].keys()) for t in tickers]
    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < MIN_OBS:
        # Fall back to union with zero-fill for missing dates
        common_dates = sorted(set.union(*date_sets))

    n_obs = len(common_dates)

    # Return matrix: {ticker: [float, ...]} aligned to common_dates
    ret_mat = {t: [rets_map[t].get(d, 0.0) for d in common_dates] for t in tickers}

    # ── 3. Portfolio weights (equity value, CASH excluded) ─────────────────
    eq_vals   = {t: stocks_data[t]["value"] for t in tickers}
    total_eq  = sum(eq_vals.values())
    weights   = (
        {t: eq_vals[t] / total_eq for t in tickers}
        if total_eq > 0 else
        {t: 1.0 / len(tickers) for t in tickers}
    )

    # ── 4. Portfolio daily returns ─────────────────────────────────────────
    port_rets = [
        sum(weights[t] * ret_mat[t][i] for t in tickers)
        for i in range(n_obs)
    ]

    # ── 5. VaR — historical simulation ────────────────────────────────────
    srt = sorted(port_rets)

    def _var(confidence):
        """Return 1-day VaR as a positive loss fraction."""
        idx = max(0, int(n_obs * (1 - confidence)) - 1)
        return -srt[idx]

    var_95_1d  = _var(0.95)
    var_99_1d  = _var(0.99)
    var_95_10d = var_95_1d  * (10 ** 0.5)
    var_99_10d = var_99_1d  * (10 ** 0.5)

    # ── 6. Per-stock & portfolio stats ────────────────────────────────────
    def _risk_stats(rets):
        n = len(rets)
        if n < 2:
            return None
        avg = sum(rets) / n
        var = sum((r - avg) ** 2 for r in rets) / (n - 1)
        std = var ** 0.5
        ann_vol = std * (252 ** 0.5)
        ann_ret = avg * 252
        sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0.0
        cum, peak, mdd = 1.0, 1.0, 0.0
        for r in rets:
            cum  *= (1 + r)
            peak  = max(peak, cum)
            mdd   = max(mdd, (peak - cum) / peak)
        return {"ann_vol": ann_vol, "ann_ret": ann_ret, "sharpe": sharpe, "max_dd": mdd}

    per_stock  = {t: _risk_stats(ret_mat[t]) for t in tickers}
    port_stats = _risk_stats(port_rets)

    # ── 7. Pearson correlation matrix ──────────────────────────────────────
    def _pearson(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        xd = [x - mx for x in xs]
        yd = [y - my for y in ys]
        num = sum(a * b for a, b in zip(xd, yd))
        den = (sum(a * a for a in xd) * sum(b * b for b in yd)) ** 0.5
        return round(num / den, 3) if den > 0 else 1.0

    corr = {
        t1: {t2: _pearson(ret_mat[t1], ret_mat[t2]) for t2 in tickers}
        for t1 in tickers
    }

    # ── 8. Build VaR banner ────────────────────────────────────────────────
    def _var_box(label, pct_loss):
        dollar = total_eq * pct_loss
        return (
            "<div style='background:var(--surface);border-radius:10px;padding:20px 24px;text-align:center;'>"
            "<div style='font-size:10px;letter-spacing:1.5px;text-transform:uppercase;"
            "color:var(--muted);margin-bottom:8px;'>{lbl}</div>"
            "<div style='font-size:24px;font-family:DM Mono,monospace;"
            "color:var(--down);font-weight:700;'>{pct:.2f}%</div>"
            "<div style='font-size:11px;color:var(--muted);margin-top:4px;'>"
            "${dlr:,.0f} on ${eq:,.0f} equity</div>"
            "</div>"
        ).format(lbl=label, pct=pct_loss * 100, dlr=dollar, eq=total_eq)

    var_section = (
        "<section>"
        "<div class='section-label'>Portfolio Value at Risk — Historical Simulation</div>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:8px;'>"
        + _var_box("95% VaR — 1 Day",   var_95_1d)
        + _var_box("99% VaR — 1 Day",   var_99_1d)
        + _var_box("95% VaR — 10 Day",  var_95_10d)
        + _var_box("99% VaR — 10 Day",  var_99_10d)
        + "</div>"
        + "<p style='font-size:11px;color:var(--muted);margin:4px 0 0 0;'>"
        + "Based on {:,} common trading days. "
          "95% VaR: worst loss exceeded on 1 in 20 days. "
          "10-day VaR = 1-day × √10 (Basel square-root-of-time scaling).".format(n_obs)
        + "</p></section>"
    )

    # ── 9. SVG correlation heatmap ────────────────────────────────────────
    n    = len(tickers)
    CELL = 54
    LM   = 85    # left margin for row labels
    TM   = 110   # top margin for column labels
    SVG_W = LM + n * CELL
    SVG_H = TM + n * CELL

    # Dark-background-friendly palette
    _NEUT = (42, 46, 66)    # neutral (≈ 0 correlation)
    _POS  = (185, 32, 32)   # strong positive = red
    _NEG  = (32, 88, 200)   # strong negative = blue

    def _corr_color(r, is_diag):
        if is_diag:
            return "rgb(58,63,88)", "#aaa"
        t  = min(1.0, abs(r))
        c1 = _NEUT
        c2 = _POS if r >= 0 else _NEG
        rc = int(c1[0] + t * (c2[0] - c1[0]))
        gc = int(c1[1] + t * (c2[1] - c1[1]))
        bc = int(c1[2] + t * (c2[2] - c1[2]))
        bright = 0.299 * rc + 0.587 * gc + 0.114 * bc
        fg = "#fff" if bright < 110 else "#bbb"
        return "rgb({},{},{})".format(rc, gc, bc), fg

    svg = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}' "
        "style='display:block;' overflow='visible'>".format(SVG_W, SVG_H)
    ]

    # Column labels (rotated +45° — text hangs above-left of anchor, clear of the grid)
    for j, t in enumerate(tickers):
        cx = LM + j * CELL + CELL // 2
        cy = TM - 8
        svg.append(
            "<text x='{cx}' y='{cy}' text-anchor='start' "
            "transform='rotate(-45,{cx},{cy})' "
            "font-family='DM Mono,monospace' font-size='11' fill='#888'>{t}</text>"
            .format(cx=cx, cy=cy, t=t)
        )

    # Row labels
    for i, t in enumerate(tickers):
        ry = TM + i * CELL + CELL // 2 + 4
        svg.append(
            "<text x='{rx}' y='{ry}' text-anchor='end' "
            "font-family='DM Mono,monospace' font-size='11' fill='#888'>{t}</text>"
            .format(rx=LM - 6, ry=ry, t=t)
        )

    # Cells
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            r_val   = corr[t1][t2]
            is_diag = (i == j)
            bg, fg  = _corr_color(r_val, is_diag)
            x = LM + j * CELL
            y = TM + i * CELL
            svg.append(
                "<rect x='{x}' y='{y}' width='{c}' height='{c}' fill='{bg}' rx='2'/>".format(
                    x=x, y=y, c=CELL - 1, bg=bg)
            )
            txt = "diag" if is_diag else "{:.2f}".format(r_val)
            if is_diag:
                txt = t1[:4]
            svg.append(
                "<text x='{cx}' y='{cy}' text-anchor='middle' dominant-baseline='middle' "
                "font-family='DM Mono,monospace' font-size='10' fill='{fg}'>{txt}</text>"
                .format(cx=x + CELL // 2, cy=y + CELL // 2, fg=fg, txt=txt)
            )

    svg.append("</svg>")
    svg_html = "".join(svg)

    legend_html = (
        "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center;"
        "margin-top:8px;font-size:11px;'>"
        "<span style='background:rgb(32,88,200);color:#fff;padding:2px 10px;"
        "border-radius:3px;'>−1.0 Negative / Hedge</span>"
        "<span style='background:rgb(42,46,66);color:#aaa;padding:2px 10px;"
        "border-radius:3px;'>0.0 Uncorrelated</span>"
        "<span style='background:rgb(185,32,32);color:#fff;padding:2px 10px;"
        "border-radius:3px;'>+1.0 Positive / Moves Together</span>"
        "<span style='color:var(--muted);'>Lower off-diagonal values = better diversification</span>"
        "</div>"
    )

    corr_section = (
        "<section style='margin-top:28px;'>"
        "<div class='section-label'>Correlation Matrix — Daily Returns</div>"
        "<div style='overflow-x:auto;'>" + svg_html + "</div>"
        + legend_html
        + "</section>"
    )

    # ── 10. Per-stock stats table ─────────────────────────────────────────
    th_s = (
        "padding:10px 14px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;"
        "color:var(--muted);border-bottom:2px solid var(--border);text-align:right;"
    )
    th_l = th_s.replace("text-align:right;", "text-align:left;")

    sorted_tickers = sorted(
        tickers,
        key=lambda t: -(per_stock[t]["sharpe"] if per_stock[t] else -99)
    )
    rows = ""
    for ticker in sorted_tickers:
        s    = per_stock[ticker]
        if not s:
            continue
        fund = stocks_data[ticker].get("fund", {})
        name = (fund.get("name") or ticker)[:24]
        w    = weights[ticker] * 100
        sh   = s["sharpe"]
        dd   = s["max_dd"]
        sh_c = "var(--up)" if sh > 1 else "var(--warn)" if sh > 0 else "var(--down)"
        dd_c = "var(--up)" if dd < 0.10 else "var(--warn)" if dd < 0.25 else "var(--down)"
        rows += (
            "<tr>"
            "<td style='padding:10px 14px;font-weight:600;'>{tkr}</td>"
            "<td style='padding:10px 14px;font-size:11px;color:var(--muted);'>{nm}</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;'>{vol:.1f}%</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "color:{shc};'>{sh:.2f}x</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "color:{ddc};'>{dd:.1f}%</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;'>{w:.1f}%</td>"
            "</tr>"
        ).format(tkr=ticker, nm=name, vol=s["ann_vol"] * 100,
                 shc=sh_c, sh=sh, ddc=dd_c, dd=dd * 100, w=w)

    if port_stats:
        ps  = port_stats
        psh = ps["sharpe"]
        pdd = ps["max_dd"]
        rows += (
            "<tr style='border-top:2px solid var(--border);'>"
            "<td style='padding:10px 14px;font-weight:700;color:var(--accent);'>PORTFOLIO</td>"
            "<td style='padding:10px 14px;font-size:11px;color:var(--muted);'>Weighted aggregate</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "font-weight:700;'>{vol:.1f}%</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "font-weight:700;color:{shc};'>{sh:.2f}x</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "font-weight:700;color:{ddc};'>{dd:.1f}%</td>"
            "<td style='padding:10px 14px;font-family:DM Mono,monospace;text-align:right;"
            "font-weight:700;'>100.0%</td>"
            "</tr>"
        ).format(
            vol=ps["ann_vol"] * 100,
            shc="var(--up)" if psh > 1 else "var(--warn)" if psh > 0 else "var(--down)",
            sh=psh,
            ddc="var(--up)" if pdd < 0.10 else "var(--warn)" if pdd < 0.25 else "var(--down)",
            dd=pdd * 100,
        )

    stats_section = (
        "<section style='margin-top:28px;'>"
        "<div class='section-label'>Per-Position Risk Statistics</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='" + th_l + "'>Ticker</th>"
        "<th style='" + th_l + "'>Company</th>"
        "<th style='" + th_s + "'>Ann. Volatility</th>"
        "<th style='" + th_s + "'>Sharpe Ratio</th>"
        "<th style='" + th_s + "'>Max Drawdown</th>"
        "<th style='" + th_s + "'>Weight</th>"
        "</tr></thead>"
        "<tbody>" + rows + "</tbody>"
        "</table></div>"
        "<p style='font-size:11px;color:var(--muted);margin:8px 0 0 0;'>"
        "Sharpe Ratio uses Rf = 4.3% annual. Ann. Vol = daily std dev × √252. "
        "Max Drawdown measured over {:,} trading days. "
        "Sorted by Sharpe descending.".format(n_obs)
        + "</p></section>"
    )

    return (
        '<div id="tab-risk" class="tab-panel">'
        '<div class="container" style="padding-top:48px;">'
        + var_section
        + corr_section
        + stats_section
        + "</div></div>"
    )


def _build_earnings_tab_html(stocks_data: dict) -> str:
    """
    Build the Earnings Calendar tab.
    Sorts all holdings by days until next earnings, soonest first.
    Stocks with no earnings date go to the bottom.
    """
    today = datetime.date.today()

    rows = []
    for ticker, sd in stocks_data.items():
        an   = sd.get("analyst", {})
        fund = sd.get("fund",   {})
        sig  = sd.get("signal", {})
        ed_str = an.get("earnings_date")
        days_to = None
        if ed_str:
            try:
                ed_date = datetime.date.fromisoformat(ed_str)
                days_to = (ed_date - today).days
            except Exception:
                ed_str = None

        fwd_eps = fund.get("fwd_eps")
        ttm_eps = fund.get("eps")
        signal  = sig.get("signal", "HOLD")
        name    = fund.get("company_name", ticker)
        price   = fund.get("price", 0) or 0

        rows.append({
            "ticker":   ticker,
            "name":     name,
            "price":    price,
            "ed_str":   ed_str,
            "days_to":  days_to,
            "fwd_eps":  fwd_eps,
            "ttm_eps":  ttm_eps,
            "signal":   signal,
        })

    # Sort: known dates by days_to asc, then unknowns
    rows.sort(key=lambda r: (r["days_to"] is None, r["days_to"] if r["days_to"] is not None else 9999))

    # Summary counts
    within_7  = sum(1 for r in rows if r["days_to"] is not None and 0 <= r["days_to"] <= 7)
    within_30 = sum(1 for r in rows if r["days_to"] is not None and 0 <= r["days_to"] <= 30)
    no_date   = sum(1 for r in rows if r["days_to"] is None)

    # ── Summary banner ────────────────────────────────────────────────────────
    banner_items = ""
    if within_7:
        banner_items += (
            "<div style='background:rgba(224,92,92,.12);border:1px solid rgba(224,92,92,.35);"
            "border-radius:8px;padding:12px 20px;text-align:center;'>"
            "<div style='font-size:28px;font-weight:700;color:#e05c5c;'>{}</div>"
            "<div style='font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;'>Reporting within 7 days</div>"
            "</div>"
        ).format(within_7)
    banner_items += (
        "<div style='background:var(--surface);border:1px solid var(--border);"
        "border-radius:8px;padding:12px 20px;text-align:center;'>"
        "<div style='font-size:28px;font-weight:700;color:var(--accent);'>{}</div>"
        "<div style='font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;'>Reporting within 30 days</div>"
        "</div>"
        "<div style='background:var(--surface);border:1px solid var(--border);"
        "border-radius:8px;padding:12px 20px;text-align:center;'>"
        "<div style='font-size:28px;font-weight:700;color:var(--text);'>{}</div>"
        "<div style='font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;'>Holdings tracked</div>"
        "</div>"
    ).format(within_30, len(rows))

    banner = (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
        "gap:12px;margin-bottom:28px;'>" + banner_items + "</div>"
    )

    # ── Table rows ────────────────────────────────────────────────────────────
    def _sig_badge(sig):
        if "BUY"  in sig: bg, clr = "rgba(0,200,150,.15)",  "#00c896"
        elif "SELL" in sig: bg, clr = "rgba(224,92,92,.15)", "#e05c5c"
        else:               bg, clr = "rgba(255,215,0,.12)",  "#ffd700"
        return (
            "<span style='background:{bg};color:{clr};font-size:9px;font-weight:700;"
            "letter-spacing:1px;padding:2px 7px;border-radius:4px;"
            "font-family:DM Mono,monospace;text-transform:uppercase;'>{s}</span>"
        ).format(bg=bg, clr=clr, s=sig or "HOLD")

    def _days_badge(days):
        if days is None:
            return "<span style='color:var(--muted);font-size:12px;'>Unknown</span>"
        if days < 0:
            return "<span style='color:var(--muted);font-size:12px;'>Past</span>"
        if days <= 7:
            bg, clr = "rgba(224,92,92,.15)", "#e05c5c"
        elif days <= 21:
            bg, clr = "rgba(240,165,0,.12)", "#f0a500"
        else:
            bg, clr = "rgba(79,142,247,.08)", "var(--accent)"
        return (
            "<span style='background:{bg};color:{clr};font-size:11px;font-weight:700;"
            "font-family:DM Mono,monospace;padding:3px 8px;border-radius:4px;'>{d}d</span>"
        ).format(bg=bg, clr=clr, d=days)

    table_rows = ""
    for r in rows:
        days   = r["days_to"]
        fwd    = "${:.2f}".format(r["fwd_eps"]) if r["fwd_eps"] else "N/A"
        ttm    = "${:.2f}".format(r["ttm_eps"]) if r["ttm_eps"] else "N/A"
        ed_disp = r["ed_str"] if r["ed_str"] else "—"
        row_bg = ""
        if days is not None and 0 <= days <= 7:
            row_bg = " style='background:rgba(224,92,92,.04);'"
        elif days is not None and 0 <= days <= 21:
            row_bg = " style='background:rgba(240,165,0,.03);'"

        table_rows += (
            "<tr{rb}>"
            "<td style='font-weight:700;color:var(--accent);font-family:DM Mono,monospace;padding:10px 12px;'>{tk}</td>"
            "<td style='padding:10px 12px;color:var(--text);'>{nm}</td>"
            "<td style='padding:10px 12px;font-family:DM Mono,monospace;'>{ed}</td>"
            "<td style='padding:10px 12px;text-align:center;'>{db}</td>"
            "<td style='padding:10px 12px;font-family:DM Mono,monospace;text-align:right;'>{fwd}</td>"
            "<td style='padding:10px 12px;font-family:DM Mono,monospace;text-align:right;color:var(--muted);'>{ttm}</td>"
            "<td style='padding:10px 12px;text-align:center;'>{sb}</td>"
            "</tr>"
        ).format(
            rb=row_bg,
            tk=r["ticker"],
            nm=r["name"][:30] + ("…" if len(r["name"]) > 30 else ""),
            ed=ed_disp,
            db=_days_badge(days),
            fwd=fwd,
            ttm=ttm,
            sb=_sig_badge(r["signal"]),
        )

    table = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='border-bottom:2px solid var(--border);'>"
        "<th style='padding:10px 12px;text-align:left;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Ticker</th>"
        "<th style='padding:10px 12px;text-align:left;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Company</th>"
        "<th style='padding:10px 12px;text-align:left;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Earnings Date</th>"
        "<th style='padding:10px 12px;text-align:center;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Days Until</th>"
        "<th style='padding:10px 12px;text-align:right;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Fwd EPS Est.</th>"
        "<th style='padding:10px 12px;text-align:right;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>TTM EPS</th>"
        "<th style='padding:10px 12px;text-align:center;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);'>Signal</th>"
        "</tr></thead>"
        "<tbody>" + table_rows + "</tbody>"
        "</table></div>"
    )

    note = ""
    if no_date:
        note = (
            "<p style='color:var(--muted);font-size:12px;margin-top:16px;'>"
            "{} holding(s) have no earnings date available from yfinance. "
            "Dates are typically available 30–60 days before the expected report.</p>"
        ).format(no_date)

    return (
        "<div id='tab-earnings' class='tab-panel'>"
        "<div class='container' style='padding-top:48px;'>"
        "<div class='section-label'>Earnings Calendar</div>"
        "<p style='color:var(--muted);font-size:13px;margin-bottom:20px;line-height:1.6;'>"
        "Holdings sorted by next earnings date. Earnings within 7 days are flagged "
        "<span style='color:#e05c5c;font-weight:600;'>red</span>, "
        "within 21 days <span style='color:#f0a500;font-weight:600;'>amber</span>. "
        "Fwd EPS is the analyst consensus estimate for next quarter.</p>"
        + banner + table + note +
        "</div></div>"
    )


def build_html(portfolio_raw: dict, stocks_data: dict, metrics: dict, ts: str) -> str:
    """Master HTML builder. Assembles all tabs."""

    # ── helper formatters ────────────────────────────────────────
    def mo(v):   return fmt_money(v)
    def pct(v):  return fmt_pct(v)
    def pcts(v): return fmt_pct(v, signed=True)
    def num(v):  return fmt_num(v)

    # ── Overview Tab ────────────────────────────────────────────────
    total_val  = metrics["total_value"]
    cash_pct   = metrics["cash_pct"]
    n_stocks   = metrics["n_stocks"]
    allocs     = metrics["allocations"]
    has_cb     = metrics.get("has_cost_basis", False)

    # sort by value descending
    alloc_sorted = sorted(
        [(t, a) for t, a in allocs.items()],
        key=lambda x: x[1]["value"], reverse=True
    )

    # counts by signal
    buys   = sum(1 for s in stocks_data.values() if "BUY"  in s.get("signal",{}).get("signal",""))
    sells  = sum(1 for s in stocks_data.values() if "SELL" in s.get("signal",{}).get("signal",""))
    holds  = n_stocks - buys - sells

    # allocation bars HTML
    alloc_bar_colors = {
        "GROWTH": "#c45aff", "VALUE": "#4f8ef7", "BLEND": "#f0a500", "CASH": "#6b7194"
    }
    alloc_html = ""
    for ticker, alloc in alloc_sorted:
        sd = stocks_data.get(ticker, {})
        ct = sd.get("classification", {}).get("type", "CASH") if ticker != "CASH" else "CASH"
        color = alloc_bar_colors.get(ct, "#4f8ef7")
        bar_pct = min(100, alloc["pct"])
        warn_marker = " ⚠" if ticker in metrics.get("concentration_warns",[]) else ""
        alloc_html += """
        <div class="alloc-bar">
          <div class="alloc-name">{tkr}{wm}</div>
          <div class="alloc-track">
            <div class="alloc-fill" style="width:{bp:.1f}%;background:{col};"></div>
          </div>
          <div class="alloc-val">{mv}  {pp}</div>
        </div>""".format(
            tkr=ticker, wm=warn_marker, bp=bar_pct, col=color,
            mv=mo(alloc["value"]), pp="{:.1f}%".format(alloc["pct"])
        )

    # Sector weight bars
    sec_weights = metrics.get("sector_weights", {})
    sec_colors  = ["#4f8ef7","#c45aff","#00c896","#f0a500","#e05c5c","#54c4d9","#a0d050","#ff7f50"]
    sec_html = ""
    for i, (sec, w) in enumerate(sorted(sec_weights.items(), key=lambda x: x[1], reverse=True)):
        col = sec_colors[i % len(sec_colors)]
        warn_sec = "⚠ " if w > 40 else ""
        sec_html += """
        <div class="alloc-bar">
          <div class="alloc-name" style="width:130px;font-size:11px;">{warn}{sec}</div>
          <div class="alloc-track">
            <div class="alloc-fill" style="width:{w:.1f}%;background:{col};"></div>
          </div>
          <div class="alloc-val">{w:.1f}%</div>
        </div>""".format(warn=warn_sec, sec=sec[:18], w=w, col=col)

    # Portfolio stat cards
    wavg_beta  = metrics.get("wavg_beta")
    wavg_growth = metrics.get("wavg_growth")
    wavg_gm    = metrics.get("wavg_gross_margin")
    wavg_up    = metrics.get("wavg_upside")
    total_pnl  = metrics.get("total_pnl")
    total_pnl_pct = metrics.get("total_pnl_pct")

    def stat_card(label, value, sub="", color="var(--text)"):
        return """<div class="stat-card">
          <div class="stat-label">{lab}</div>
          <div class="stat-value" style="font-size:24px;color:{col};">{val}</div>
          {sub_html}
        </div>""".format(lab=label, val=value, col=color,
            sub_html='<div class="stat-sub">{}</div>'.format(sub) if sub else "")

    stat_cards_html = stat_card("Total Value", mo(total_val))
    if has_cb and total_pnl is not None:
        pnl_col = "var(--up)" if total_pnl >= 0 else "var(--down)"
        stat_cards_html += stat_card("Unrealized P&L",
            "{}{:.1f}%".format("+" if total_pnl_pct>=0 else "", total_pnl_pct or 0),
            mo(total_pnl), pnl_col)
    stat_cards_html += stat_card("Cash",
        "{:.1f}%".format(cash_pct), mo(metrics["cash"]),
        "var(--warn)" if cash_pct > 20 else "var(--up)")
    if wavg_beta:
        bcol = "var(--down)" if wavg_beta > 1.5 else "var(--warn)" if wavg_beta > 1.1 else "var(--up)"
        stat_cards_html += stat_card("Portfolio Beta", "{:.2f}".format(wavg_beta), "vs S&P 500", bcol)
    if wavg_up is not None:
        ucol = "var(--up)" if wavg_up > 0 else "var(--down)"
        stat_cards_html += stat_card("Wtd-Avg Upside", "{:+.1f}%".format(wavg_up), "to fair value", ucol)
    if wavg_growth:
        stat_cards_html += stat_card("Wtd-Avg Growth", "{:.0f}%".format(wavg_growth*100), "est. revenue")
    stat_cards_html += stat_card("Buy Signals", str(buys), color="var(--up)")
    stat_cards_html += stat_card("Sell Signals", str(sells), color="var(--down)")

    # Overview portfolio table — with optional P&L columns
    pnl_header = "<th>P&L</th><th>P&L %</th>" if has_cb else ""
    overview_rows = ""
    for ticker, alloc in alloc_sorted:
        if ticker == "CASH":
            pnl_cells = "<td>—</td><td>—</td>" if has_cb else ""
            overview_rows += """
            <tr>
              <td><strong>CASH</strong></td>
              <td style="color:var(--muted);">—</td>
              <td>—</td>
              <td style="font-family:'DM Mono',monospace;">{val}</td>
              <td>{pct_val:.1f}%</td>
              {pnl_cells}
              <td>—</td>
              <td>—</td>
            </tr>""".format(val=mo(alloc["value"]), pct_val=alloc["pct"],
                            pnl_cells=pnl_cells)
            continue
        sd   = stocks_data.get(ticker, {})
        fund = sd.get("fund", {})
        sig  = sd.get("signal", {})
        cls  = sd.get("classification", {})
        signal_str = sig.get("signal", "—")
        sig_color  = signal_color(signal_str)
        sig_bg_col = signal_bg(signal_str)
        t_color    = type_color(cls.get("type",""))
        price      = fund.get("price")
        best_fv    = sig.get("best_fv")
        upside_val = ((best_fv - price) / price * 100) if (best_fv and price) else None
        upside_str = pcts(upside_val) if upside_val is not None else "—"
        upside_col = "var(--up)" if (upside_val and upside_val > 0) else "var(--down)" if upside_val else "var(--muted)"

        pnl_cells = ""
        if has_cb:
            pnl_v    = alloc.get("pnl")
            pnl_p    = alloc.get("pnl_pct")
            if pnl_v is not None:
                pc = "var(--up)" if pnl_v >= 0 else "var(--down)"
                pnl_cells = "<td style='color:{pc};'>{pv}</td><td style='color:{pc};'>{pp}</td>".format(
                    pc=pc, pv=mo(pnl_v), pp="{:+.1f}%".format(pnl_p or 0))
            else:
                pnl_cells = "<td style='color:var(--muted);'>—</td><td style='color:var(--muted);'>—</td>"

        # earnings date warning
        an = sd.get("analyst", {})
        ed = an.get("earnings_date")
        ed_flag = ""
        if ed:
            try:
                days_to = (datetime.date.fromisoformat(ed) - datetime.date.today()).days
                if 0 <= days_to <= 14:
                    ed_flag = " <span style='font-size:9px;color:var(--warn);font-family:DM Mono,monospace;'>⚡ ERN {}</span>".format(ed)
            except Exception: pass

        overview_rows += """
        <tr>
          <td><strong style="color:var(--accent);">{tkr}</strong>{ed}</td>
          <td style="font-size:12px;color:var(--muted);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</td>
          <td><span class="type-badge" style="color:{tcol};border-color:{tcol};background:rgba(79,142,247,0.07);">{ctype}</span></td>
          <td>{val}</td>
          <td>{pct_val:.1f}%</td>
          {pnl_cells}
          <td style="color:{ucol};">{upsid}</td>
          <td><span class="signal-pill" style="color:{scol};background:{sbg};">{sig}</span></td>
        </tr>""".format(
            tkr=ticker, ed=ed_flag, name=fund.get("company_name", ticker)[:28],
            tcol=t_color, ctype=cls.get("type","—"),
            val=mo(alloc["value"]), pct_val=alloc["pct"],
            pnl_cells=pnl_cells,
            ucol=upside_col, upsid=upside_str,
            scol=sig_color, sbg=sig_bg_col, sig=signal_str
        )

    # concentration warnings + sector clustering warning
    conc_html = ""
    for w in metrics.get("concentration_warns", []):
        pct_v = allocs.get(w, {}).get("pct", 0)
        conc_html += '<div class="danger-banner"><strong>⚠ CONCENTRATION RISK: {}</strong> — {:.1f}% of portfolio exceeds the {:.0f}% maximum. Consider trimming.</div>'.format(
            w, pct_v, MAX_POSITION_PCT * 100)
    for w in metrics.get("trim_warns", []):
        pct_v = allocs.get(w, {}).get("pct", 0)
        conc_html += '<div class="warn-banner"><strong>⚡ TRIM MONITOR: {}</strong> — {:.1f}% position approaching concentration limit.</div>'.format(w, pct_v)
    for sec, tickers in metrics.get("concentration_sectors", {}).items():
        w = sec_weights.get(sec, 0)
        conc_html += '<div class="warn-banner"><strong>📊 SECTOR CONCENTRATION: {}</strong> — {:.0f}% in {} positions ({}). High correlation risk — moves together.</div>'.format(
            sec, w, len(tickers), ", ".join(tickers))

    overview_tab = """
<div id="tab-overview" class="tab-panel active">
<div class="hero">
  <div class="hero-label">Portfolio Analysis Report</div>
  <h1 class="hero-title">Portfolio Dashboard</h1>
  <p class="hero-subtitle">Multi-method signals · {ts}</p>
</div>
<div class="container">
  {conc}
  <section>
    <div class="stat-grid">{stats}</div>
  </section>
  <section>
    <div class="section-label">Allocation by Position</div>
    {alloc_bars}
  </section>
  <section>
    <div class="section-label">Sector Breakdown</div>
    {sec_bars}
  </section>
  <section>
    <div class="section-label">Holdings Summary</div>
    <div style="overflow-x:auto;">
    <table class="port-table">
      <thead><tr>
        <th>Ticker</th><th>Name</th><th>Type</th>
        <th>Value</th><th>Weight</th>
        {pnl_hdr}
        <th>Best FV Upside</th><th>Signal</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    {pnl_note}
  </section>
</div>
</div>""".format(
        ts=ts, conc=conc_html,
        stats=stat_cards_html,
        alloc_bars=alloc_html,
        sec_bars=sec_html,
        pnl_hdr=pnl_header,
        pnl_note='<div style="font-size:11px;color:var(--muted);margin-top:8px;">⚡ = Earnings within 14 days. Add cost_basis column to CSV to see P&L.</div>' if not has_cb else '',
        rows=overview_rows
    )

    # ── Signals Tab ────────────────────────────────────────────────
    signals_cards = ""
    for ticker in sorted(stocks_data.keys()):
        sd   = stocks_data[ticker]
        fund = sd.get("fund", {})
        sig  = sd.get("signal", {})
        cls  = sd.get("classification", {})
        tech = sd.get("tech", {})

        signal_str = sig.get("signal", "—")
        strength   = sig.get("strength", 50)
        sig_col    = signal_color(signal_str)
        sig_bg_col = signal_bg(signal_str)
        t_col      = type_color(cls.get("type", "BLEND"))

        price    = fund.get("price")
        best_fv  = sig.get("best_fv")
        best_m   = sig.get("best_method", "—")
        all_fvs  = sig.get("all_fvs", {})

        # FV table rows
        fv_rows = ""
        for method, fv in sorted(all_fvs.items(), key=lambda x: x[1] or 0, reverse=True):
            if fv and price:
                upside_v = (fv - price) / price * 100
                u_col = "var(--up)" if upside_v > 0 else "var(--down)"
                best_marker = " ★" if method == best_m else ""
                fv_rows += "<tr><td style='color:var(--muted);font-size:12px;padding:4px 0;'>{m}{bm}</td><td style='font-family:DM Mono,monospace;font-size:12px;text-align:right;'>{fv}</td><td style='font-family:DM Mono,monospace;font-size:12px;text-align:right;color:{uc};'>{up}</td></tr>".format(
                    m=method, bm=best_marker, fv=mo(fv), up=pcts(upside_v), uc=u_col)

        # Reasons
        buy_reasons  = sig.get("reasons",{}).get("buy",[])
        sell_reasons = sig.get("reasons",{}).get("sell",[])
        hold_reasons = sig.get("reasons",{}).get("hold",[])

        buy_li  = "".join("<li>{}</li>".format(r) for r in buy_reasons)
        sell_li = "".join("<li>{}</li>".format(r) for r in sell_reasons)
        hold_li = "".join("<li>{}</li>".format(r) for r in hold_reasons)

        reasons_html = ""
        if buy_li:
            reasons_html += '<div style="margin-bottom:10px;"><div style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--up);margin-bottom:4px;">Buy Signals</div><ul class="reason-list">{}</ul></div>'.format(buy_li)
        if sell_li:
            reasons_html += '<div style="margin-bottom:10px;"><div style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--down);margin-bottom:4px;">Sell Signals</div><ul class="reason-list sell">{}</ul></div>'.format(sell_li)
        if hold_li:
            reasons_html += '<div style="margin-bottom:10px;"><div style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--warn);margin-bottom:4px;">Hold Notes</div><ul class="reason-list hold">{}</ul></div>'.format(hold_li)

        # Sparkline
        sparkline = mini_sparkline(tech.get("closes_60", []), color=sig_col)

        rsi_val = tech.get("rsi")
        rsi_disp = "{:.0f}".format(rsi_val) if rsi_val else "—"
        rsi_col  = rsi_color(rsi_val)

        # Build context chips
        an_s      = sd.get("analyst", {})
        chips_html = ""
        if an_s.get("dividend_yield"):
            chips_html += '<span style="font-size:10px;background:rgba(0,200,150,0.12);color:var(--up);border:1px solid rgba(0,200,150,0.25);border-radius:4px;padding:2px 8px;margin-right:4px;">💰 {:.2f}% yield</span>'.format(an_s["dividend_yield"])
        spf = an_s.get("short_pct_float")
        if spf and spf > 5:
            sc_col = "rgba(224,92,92,0.15)" if spf > 15 else "rgba(240,165,0,0.12)"
            sc_bcol = "rgba(224,92,92,0.4)" if spf > 15 else "rgba(240,165,0,0.3)"
            sc_tcol = "var(--down)" if spf > 15 else "var(--warn)"
            chips_html += '<span style="font-size:10px;background:{};color:{};border:1px solid {};border-radius:4px;padding:2px 8px;margin-right:4px;">🩳 {:.1f}% short</span>'.format(sc_col, sc_tcol, sc_bcol, spf)
        _ed2 = an_s.get("earnings_date")
        if _ed2:
            try:
                _d2 = (datetime.date.fromisoformat(_ed2) - datetime.date.today()).days
                if 0 <= _d2 <= 21:
                    chips_html += '<span style="font-size:10px;background:rgba(240,165,0,0.12);color:var(--warn);border:1px solid rgba(240,165,0,0.3);border-radius:4px;padding:2px 8px;margin-right:4px;">⚡ Earnings in {}d</span>'.format(_d2)
                elif _d2 > 0:
                    chips_html += '<span style="font-size:10px;background:rgba(107,113,148,0.12);color:var(--muted);border:1px solid rgba(107,113,148,0.2);border-radius:4px;padding:2px 8px;margin-right:4px;">📅 Earnings {}</span>'.format(_ed2)
            except Exception: pass
        p52 = tech.get("pct_from_52wk_high")
        if p52 is not None:
            p52_col = "var(--up)" if p52 >= -5 else "var(--down)" if p52 <= -30 else "var(--muted)"
            chips_html += '<span style="font-size:10px;background:rgba(107,113,148,0.1);color:{};border:1px solid rgba(107,113,148,0.2);border-radius:4px;padding:2px 8px;margin-right:4px;">{:+.1f}% vs 52wk high</span>'.format(p52_col, p52)

        signals_cards += """
        <div class="stock-card">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
              <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="type-badge" style="color:{tcol};border-color:{tcol};">{ctype}</span>
                <span style="font-size:11px;color:var(--muted);">{sector}</span>
              </div>
              <div style="margin-top:8px;">{chips}</div>
            </div>
            <div style="margin-left:24px;flex:1;">{sparkline}</div>
            <div class="sc-price">
              <div class="sc-price-val">{price_disp}</div>
              <div style="margin-top:8px;"><span class="signal-pill" style="color:{scol};background:{sbg};font-size:11px;">{sig}</span></div>
              <div style="margin-top:8px;font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);">Strength {st}%</div>
              <div class="strength-bar"><div class="strength-fill" style="width:{st}%;background:{scol};"></div></div>
            </div>
          </div>
          <div class="stock-card-body">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;flex-wrap:wrap;">
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;">Valuation Methods</div>
                <table style="width:100%;border-collapse:collapse;">
                  <thead><tr>
                    <th style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:4px 0;text-align:left;">Method</th>
                    <th style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:4px 0;text-align:right;">Fair Value</th>
                    <th style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:4px 0;text-align:right;">Upside</th>
                  </tr></thead>
                  <tbody>{fv_rows}</tbody>
                </table>
                <div style="font-size:10px;color:var(--muted);margin-top:8px;">★ = Best backtested method</div>
              </div>
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;">Signal Breakdown</div>
                {reasons}
              </div>
            </div>
          </div>
        </div>""".format(
            tkr=ticker,
            name=fund.get("company_name", ticker),
            tcol=t_col, ctype=cls.get("type","—"),
            sector=fund.get("industry") or fund.get("sector") or "",
            chips=chips_html,
            sparkline=sparkline,
            price_disp=mo(price) if price else "—",
            scol=sig_col, sbg=sig_bg_col,
            sig=signal_str, st=strength,
            fv_rows=fv_rows, reasons=reasons_html,
            rsi_disp=rsi_disp, rsi_col=rsi_col,
        )

    signals_tab = """
<div id="tab-signals" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Buy / Hold / Sell Signals — All Holdings</div>
  {cards}
</div>
</div>""".format(cards=signals_cards)

    # ── Technical Analysis Tab ─────────────────────────────────────
    tech_rows_html = ""
    for ticker in sorted(stocks_data.keys()):
        sd   = stocks_data[ticker]
        tech = sd.get("tech", {})
        fund = sd.get("fund", {})
        sig  = sd.get("signal", {})

        if not tech:
            continue

        rsi     = tech.get("rsi")
        ema10   = tech.get("ema10")
        ema50   = tech.get("ema50")
        sma200  = tech.get("sma200")
        price   = tech.get("price") or fund.get("price")
        bb      = tech.get("bb", {})
        macd_d  = tech.get("macd", {})
        pct200  = tech.get("pct_above_sma200")

        def ind_row(label, value, signal_text, icon, color):
            return """<div class="tech-row">
              <div class="tech-indicator">{lab}</div>
              <div class="tech-value" style="color:{col};">{val}</div>
              <div class="tech-signal">{sig}</div>
              <div class="tech-icon">{icon}</div>
            </div>""".format(lab=label, val=value, sig=signal_text, icon=icon, col=color)

        rows = ""

        # RSI
        if rsi is not None:
            if rsi < 35:
                rsig, rico, ricon = "Oversold — potential buy zone", "var(--up)", "🟢"
            elif rsi < 45:
                rsig, rico, ricon = "Pullback zone — entry opportunity", "var(--up)", "🟢"
            elif rsi > 70:
                rsig, rico, ricon = "Overbought — consider trimming", "var(--down)", "🔴"
            elif rsi > 60:
                rsig, rico, ricon = "Extended — watch for exhaustion", "var(--warn)", "🟡"
            else:
                rsig, rico, ricon = "Neutral momentum", "var(--muted)", "⚪"
            rows += ind_row("RSI (14)", "{:.1f}".format(rsi), rsig, ricon, rico)

        # EMAs
        if ema10 and price:
            diff = (price - ema10) / ema10 * 100
            if abs(diff) < 3:
                esig, ecol, eico = "At 10-EMA support/resistance", "var(--warn)", "🟡"
            elif diff > 0:
                esig, ecol, eico = "{:+.1f}% above 10-EMA".format(diff), "var(--up)", "🟢"
            else:
                esig, ecol, eico = "{:+.1f}% below 10-EMA".format(diff), "var(--down)", "🔴"
            rows += ind_row("10-Day EMA", "${:.2f}".format(ema10), esig, eico, ecol)

        if ema50 and price:
            diff = (price - ema50) / ema50 * 100
            if abs(diff) < 3:
                esig, ecol, eico = "At 50-EMA — classic pullback entry", "var(--up)", "🟢"
            elif diff > 0:
                esig, ecol, eico = "{:+.1f}% above 50-EMA".format(diff), "var(--up)", "🟢"
            else:
                esig, ecol, eico = "{:+.1f}% below 50-EMA — caution", "var(--down)", "🔴"
            rows += ind_row("50-Day EMA", "${:.2f}".format(ema50), esig, eico, ecol)

        if sma200 and price:
            marker = "BELOW 200-SMA — TREND BREAK" if tech.get("below_sma200") else "{:+.1f}% above 200-SMA".format(pct200 or 0)
            s200col = "var(--down)" if tech.get("below_sma200") else "var(--up)"
            s200ico = "🔴" if tech.get("below_sma200") else "🟢"
            rows += ind_row("200-Day SMA", "${:.2f}".format(sma200), marker, s200ico, s200col)

        # MACD
        if macd_d.get("macd") is not None:
            hist_v = macd_d.get("hist", 0) or 0
            if hist_v > 0:
                msig, mcol, mico = "MACD above signal — bullish momentum", "var(--up)", "🟢"
            else:
                msig, mcol, mico = "MACD below signal — bearish momentum", "var(--down)", "🔴"
            if tech.get("macd_div"):
                msig, mcol, mico = "⚠ BEARISH DIVERGENCE DETECTED", "var(--down)", "🔴"
            rows += ind_row("MACD", "{:.3f}".format(macd_d["macd"]), msig, mico, mcol)

        # Bollinger
        if bb.get("upper"):
            pct_b = bb.get("pct_b", 0.5)
            if pct_b > 0.95:
                bsig, bcol, bico = "Near upper band — overbought risk", "var(--down)", "🔴"
                if tech.get("bb_fail"):
                    bsig, bcol, bico = "⚠ TOUCH-AND-FAIL signal", "var(--down)", "🔴"
            elif pct_b < 0.10:
                bsig, bcol, bico = "Near lower band — potential reversal", "var(--up)", "🟢"
            else:
                bsig, bcol, bico = "%B={:.0f}% — mid-band range".format(pct_b*100), "var(--muted)", "⚪"
            rows += ind_row("Bollinger %B", "{:.0f}%".format(pct_b*100), bsig, bico, bcol)

        # Stochastic RSI
        srsi = tech.get("stoch_rsi", {})
        if srsi:
            sk = srsi.get("k")
            if srsi.get("oversold"):
                srsig, srcol, srico = "Oversold (<20) — short-term buy zone", "var(--up)", "🟢"
            elif srsi.get("overbought"):
                srsig, srcol, srico = "Overbought (>80) — short-term sell risk", "var(--down)", "🔴"
            else:
                srsig, srcol, srico = "Neutral ({:.0f})".format(sk or 50), "var(--muted)", "⚪"
            rows += ind_row("Stoch RSI %K", "{:.0f}".format(sk) if sk is not None else "—", srsig, srico, srcol)

        # ATR
        atr = tech.get("atr")
        if atr and price:
            atr_pct = atr / price * 100
            rows += ind_row("ATR (14)", "${:.2f}  ({:.1f}%)".format(atr, atr_pct),
                            "Daily volatility — use for stop placement", "⚪", "var(--muted)")

        # 52-week range
        w52h = tech.get("week52_high")
        w52l = tech.get("week52_low")
        pct52 = tech.get("pct_from_52wk_high")
        if w52h and w52l and price:
            if tech.get("near_52wk_high"):
                w52sig, w52col, w52ico = "Near 52-week high — strong momentum", "var(--up)", "🟢"
            elif tech.get("far_from_52wk_high"):
                w52sig, w52col, w52ico = "{:.0f}% below 52-week high — significant drawdown".format(abs(pct52 or 0)), "var(--down)", "🔴"
            else:
                w52sig, w52col, w52ico = "{:.1f}% below 52-week high".format(abs(pct52 or 0)), "var(--muted)", "⚪"
            rows += ind_row("52-Week Range", "${:.0f} – ${:.0f}".format(w52l, w52h), w52sig, w52ico, w52col)

        # Volume
        if tech.get("high_volume"):
            rows += ind_row("Volume", "HIGH", "1.5× above 20-day average — institutional flow", "🟢", "var(--up)")

        sig_str  = sig.get("signal", "—")
        sig_col2 = signal_color(sig_str)

        tech_rows_html += """
        <div class="stock-card" style="margin-bottom:20px;">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-family:'DM Mono',monospace;font-size:18px;font-weight:700;">{price_d}</div>
              <div style="margin-top:6px;"><span class="signal-pill" style="color:{scol};background:{sbg};font-size:10px;">{sig_d}</span></div>
            </div>
          </div>
          <div class="stock-card-body">{rows}</div>
        </div>""".format(
            tkr=ticker, name=fund.get("company_name", ticker),
            price_d=mo(price) if price else "—",
            scol=sig_col2, sbg=signal_bg(sig_str), sig_d=sig_str,
            rows=rows
        )

    tech_tab = """
<div id="tab-technical" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Technical Indicators — All Holdings</div>
  {rows}
</div>
</div>""".format(rows=tech_rows_html)

    # ── Fundamentals Tab ───────────────────────────────────────────
    fund_html = ""
    for ticker in sorted(stocks_data.keys()):
        sd   = stocks_data[ticker]
        fund = sd.get("fund", {})
        cls  = sd.get("classification", {})
        sig  = sd.get("signal", {})

        if not fund:
            continue

        rev_g = (fund.get("rev_growth") or 0) * 100
        gm    = fund.get("gross_margin") or 0
        ro40  = fund.get("ro40") or 0
        fcf_m = fund.get("fcf_margin") or 0
        op_m  = (fund.get("op_margin") or 0) * 100

        # Health scores for each metric
        def health_badge(val, good, ok):
            if val >= good: return 'style="color:var(--up);"', "▲ Strong"
            if val >= ok:   return 'style="color:var(--warn);"', "● OK"
            return 'style="color:var(--down);"', "▼ Weak"

        rev_style, rev_label = health_badge(rev_g, 20, 10)
        gm_style,  gm_label  = health_badge(gm,    60, 40)
        ro40_style, ro40_lbl = health_badge(ro40,   40, 20)
        fcf_style,  fcf_lbl  = health_badge(fcf_m,  15, 5)

        # Story check
        story_flags = []
        if rev_g < 10 and cls.get("type") == "GROWTH":
            story_flags.append("⚠ Revenue growth below 10% — growth story decelerating")
        if gm < 40 and cls.get("type") == "GROWTH":
            story_flags.append("⚠ Gross margin below 40% — pricing power concerns")
        if ro40 < 20:
            story_flags.append("⚠ Rule-of-40 below 20 — efficiency concern")
        if ro40 >= 40:
            story_flags.append("✓ Rule-of-40 above 40 — excellent efficiency")

        flags_html = ""
        for fl in story_flags:
            col = "var(--up)" if fl.startswith("✓") else "var(--warn)"
            flags_html += '<div style="font-size:12px;color:{};padding:4px 0;">{}  </div>'.format(col, fl)

        t_col  = type_color(cls.get("type","BLEND"))
        an_fd  = sd.get("analyst", {})
        # Earnings imminence flag
        _earn_soon = False
        _ed = an_fd.get("earnings_date")
        if _ed:
            try:
                _days = (datetime.date.fromisoformat(_ed) - datetime.date.today()).days
                _earn_soon = 0 <= _days <= 14
            except Exception: pass

        fund_html += """
        <div class="stock-card" style="margin-bottom:20px;">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
            </div>
            <div style="margin-left:auto;display:flex;gap:10px;align-items:center;">
              <span class="type-badge" style="color:{tcol};border-color:{tcol};">{ctype}</span>
              <span style="font-size:12px;color:var(--muted);">{sector}</span>
            </div>
          </div>
          <div class="stock-card-body">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Key Metrics</div>
                <div class="fund-block">
                  <div class="fund-row"><span class="fund-key">Revenue Growth (TTM)</span><span class="fund-val" {rstyle}>{rev_g:.1f}% {rl}</span></div>
                  <div class="fund-row"><span class="fund-key">Gross Margin</span><span class="fund-val" {gmstyle}>{gm:.1f}% {gml}</span></div>
                  <div class="fund-row"><span class="fund-key">Operating Margin</span><span class="fund-val">{op_m:.1f}%</span></div>
                  <div class="fund-row"><span class="fund-key">FCF Margin</span><span class="fund-val" {fcfstyle}>{fcf_m:.1f}% {fcfl}</span></div>
                  <div class="fund-row"><span class="fund-key">Rule of 40</span><span class="fund-val" {ro40style}>{ro40:.1f} {ro40l}</span></div>
                  <div class="fund-row"><span class="fund-key">Market Cap</span><span class="fund-val">{mktcap}</span></div>
                  <div class="fund-row"><span class="fund-key">P/E Ratio</span><span class="fund-val">{pe}</span></div>
                  <div class="fund-row"><span class="fund-key">EV/Revenue</span><span class="fund-val">{evr}</span></div>
                  <div class="fund-row"><span class="fund-key">Dividend Yield</span><span class="fund-val">{div_yield}</span></div>
                  <div class="fund-row"><span class="fund-key">Short % of Float</span><span class="fund-val" {short_col}>{short_float}</span></div>
                  <div class="fund-row"><span class="fund-key">Short Ratio (Days)</span><span class="fund-val">{short_ratio}</span></div>
                  <div class="fund-row"><span class="fund-key">Next Earnings</span><span class="fund-val" {earn_col}>{earn_date}</span></div>
                  <div class="fund-row"><span class="fund-key">Fwd EPS (NTM)</span><span class="fund-val">{fwd_eps_disp}</span></div>
                  <div class="fund-row"><span class="fund-key">Fwd Revenue (NTM)</span><span class="fund-val">{fwd_rev_disp}</span></div>
                </div>
              </div>
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Fundamental Health Check</div>
                {flags}
                <div style="margin-top:16px;">
                  <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;">Classification Rationale</div>
                  <ul class="reason-list">
                    {cls_reasons}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>""".format(
            tkr=ticker, name=fund.get("company_name", ticker),
            tcol=t_col, ctype=cls.get("type","—"),
            sector=fund.get("industry") or fund.get("sector") or "",
            rstyle=rev_style, rev_g=rev_g, rl=rev_label,
            gmstyle=gm_style, gm=gm, gml=gm_label,
            op_m=op_m,
            fcfstyle=fcf_style, fcf_m=fcf_m, fcfl=fcf_lbl,
            ro40style=ro40_style, ro40=ro40, ro40l=ro40_lbl,
            mktcap=mo(fund.get("market_cap")),
            pe="{:.1f}x".format(fund.get("pe")) if fund.get("pe") else "—",
            div_yield=("{:.2f}%".format(an_fd.get("dividend_yield")) if an_fd.get("dividend_yield") else "—"),
            short_col=('style="color:var(--warn);"' if (an_fd.get("short_pct_float") or 0) > 10 else ''),
            short_float=("{:.1f}%".format(an_fd.get("short_pct_float")) if an_fd.get("short_pct_float") else "—"),
            short_ratio=("{:.1f}d".format(an_fd.get("short_ratio")) if an_fd.get("short_ratio") else "—"),
            earn_col=('style="color:var(--warn);font-weight:700;"' if _earn_soon else ''),
            earn_date=an_fd.get("earnings_date") or "—",
            beta_val=("{:.2f}".format(fund.get("beta")) if fund.get("beta") else "—"),
            fwd_eps_disp=("${:.2f} ({})".format(fund.get("fwd_eps"), fund.get("fwd_eps_source","")) if fund.get("fwd_eps") else "—"),
            fwd_rev_disp=("${:.1f}B ({})".format(fund.get("fwd_rev",0)/1e9, fund.get("fwd_rev_source","")) if fund.get("fwd_rev") else "—"),
            evr="{:.1f}x".format(fund.get("ev_rev")) if fund.get("ev_rev") else "—",
            flags=flags_html,
            cls_reasons="".join("<li>{}</li>".format(r) for r in cls.get("reasons",[])[:4]),
        )

    fund_tab = """
<div id="tab-fundamentals" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Fundamental Health — All Holdings</div>
  {content}
</div>
</div>""".format(content=fund_html)

    # ── Earnings Calendar Tab ──────────────────────────────────────
    earnings_tab = _build_earnings_tab_html(stocks_data)

    # ── Backtest Tab ───────────────────────────────────────────────
    bt_rows = ""
    for ticker in sorted(stocks_data.keys()):
        sd   = stocks_data[ticker]
        bt   = sd.get("backtest", {})
        fund = sd.get("fund", {})
        if not bt or not bt.get("best_method"):
            continue
        best     = bt["best_method"]
        mapes    = bt.get("mape", {})
        rel      = bt.get("relative_score", {})
        fv_by_m  = bt.get("fv", {})
        price    = fund.get("price") or 0

        # Sort methods: best (lowest MAPE) first
        all_methods = sorted(mapes.keys(), key=lambda m: mapes[m])

        # Build one column per method
        header_cells = ""
        mape_cells   = ""
        rel_cells    = ""
        fv_cells     = ""
        upside_cells = ""

        for method in all_methods:
            is_best  = (method == best)
            th_style = "font-family:DM Mono,monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;padding:10px 14px;text-align:right;border-bottom:2px solid var(--border);"
            th_color = "color:var(--accent);" if is_best else "color:var(--muted);"
            td_base  = "font-family:DM Mono,monospace;font-size:12px;text-align:right;padding:8px 14px;border-top:1px solid var(--border);"
            td_best  = "font-weight:700;color:var(--accent);" if is_best else "color:var(--text);"
            td_muted = "color:var(--muted);"

            mape_val = mapes.get(method, 0)
            rel_val  = rel.get(method, 0)
            fv_val   = fv_by_m.get(method)
            upside   = ((fv_val - price) / price * 100) if (fv_val and price > 0) else None
            u_color  = "color:var(--up);" if (upside and upside > 0) else "color:var(--down);" if upside else "color:var(--muted);"

            star = " ★" if is_best else ""
            header_cells  += "<th style='{}{}' >{}{}</th>".format(th_style, th_color, method, star)
            mape_cells    += "<td style='{}{}'>{:.1f}%</td>".format(td_base, td_muted, mape_val)
            rel_cells     += "<td style='{}{}'>{:.0f}</td>".format(td_base, td_best, rel_val)
            fv_cells      += "<td style='{}{}'>{}</td>".format(td_base, td_best if is_best else td_muted, mo(fv_val) if fv_val else "—")
            upside_cells  += "<td style='{}{}' >{}</td>".format(td_base, u_color, "{:+.0f}%".format(upside) if upside is not None else "—")

        row_label_style = "font-family:DM Mono,monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:8px 14px;border-top:1px solid var(--border);white-space:nowrap;"

        bt_rows += """
        <div class="stock-card" style="margin-bottom:20px;">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">Best Method</div>
              <div style="font-family:'DM Mono',monospace;font-size:18px;font-weight:700;color:var(--accent);">{best} ★</div>
              <div style="font-size:11px;color:var(--muted);margin-top:4px;">MAPE {best_mape:.1f}% · Score {best_rel:.0f}/100</div>
            </div>
          </div>
          <div style="overflow-x:auto;padding:0 24px 20px;">
            <table style="width:100%;border-collapse:collapse;min-width:500px;">
              <thead>
                <tr>
                  <th style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:10px 14px;text-align:left;border-bottom:2px solid var(--border);">Metric</th>
                  {hcells}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style="{rl_style}">MAPE <span style="font-size:9px;">(lower = better)</span></td>
                  {mape_cells}
                </tr>
                <tr>
                  <td style="{rl_style}">Relative Score <span style="font-size:9px;">(higher = better, best=100)</span></td>
                  {rel_cells}
                </tr>
                <tr>
                  <td style="{rl_style}">Fair Value</td>
                  {fv_cells}
                </tr>
                <tr>
                  <td style="{rl_style}">Upside vs Price</td>
                  {upside_cells}
                </tr>
              </tbody>
            </table>
          </div>
        </div>""".format(
            tkr=ticker, name=fund.get("company_name", ticker),
            best=best,
            best_mape=mapes.get(best, 0),
            best_rel=rel.get(best, 100),
            hcells=header_cells,
            mape_cells=mape_cells,
            rel_cells=rel_cells,
            fv_cells=fv_cells,
            upside_cells=upside_cells,
            rl_style=row_label_style,
        )

    bt_tab = """
<div id="tab-backtest" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Backtesting — Best Valuation Method Per Stock</div>
  <div class="warn-banner" style="margin-bottom:24px;">
    <strong>How to read this table:</strong><br>
    <strong>MAPE</strong> (Mean Absolute Percentage Error) — average % gap between each method's static fair value and the actual daily closing price over the backtest window. <em>Lower is better.</em><br>
    <strong>Relative Score</strong> — ranks methods against each other on a 0–100 scale. The best method always scores 100; others are scored proportionally. This avoids the "0% everywhere" problem that occurs with growth stocks trading at large premiums to intrinsic value.<br>
    <strong>★ Best Method</strong> — the one with the lowest MAPE. Its fair value is used as the primary signal in the Signals tab.<br>
    Prices from yfinance · Fundamentals from TradingView.
  </div>
  {rows}
</div>
</div>""".format(rows=bt_rows or '<p style="color:var(--muted);">No backtest data available — price history may be insufficient.</p>')

    # ── Rebalance Tab ──────────────────────────────────────────────
    rebal_actions = []

    # 1. Concentration trims — with exact dollar/share target
    for ticker in metrics.get("concentration_warns", []):
        a    = allocs.get(ticker, {})
        sd   = stocks_data.get(ticker, {})
        pr   = (sd.get("fund",{}) or {}).get("price") or 0
        sh   = a.get("shares", 0)
        cur_pct = a.get("pct", 0)
        trim = trim_to_target(sh, pr, cur_pct, MAX_POSITION_PCT * 100, total_val)
        trade_detail = ""
        if trim:
            trade_detail = " → Sell ~{} shares (${:,.0f}) to reach {:.0f}% target.".format(
                trim["shares"], trim["dollars"], MAX_POSITION_PCT * 100)
        rebal_actions.append({
            "type": "TRIM", "ticker": ticker, "priority": 1,
            "reason": "Position is {:.1f}% of portfolio (max {:.0f}%).{}".format(
                cur_pct, MAX_POSITION_PCT * 100, trade_detail),
            "color": "var(--down)"
        })

    # 2. Sell signals — with ATR stop reference
    for ticker, sd in stocks_data.items():
        sig  = sd.get("signal", {})
        fund = sd.get("fund", {})
        tech = sd.get("tech", {})
        if "SELL" in sig.get("signal", ""):
            if ticker not in [r["ticker"] for r in rebal_actions]:
                atr_v = tech.get("atr")
                pr    = fund.get("price") or 0
                stop_note = ""
                if atr_v and pr:
                    stop_note = "  ATR stop: ${:.2f} ({:.1f}% below price).".format(
                        pr - 2*atr_v, 2*atr_v/pr*100)
                rebal_actions.append({
                    "type": "SELL", "ticker": ticker, "priority": 2,
                    "reason": "  ".join(sig.get("reasons",{}).get("sell",[])[:2]) + stop_note,
                    "color": "var(--down)"
                })

    # 3. Earnings-imminent warning (don't open new positions 2 weeks before earnings)
    for ticker, sd in stocks_data.items():
        an_r = sd.get("analyst", {})
        ed   = an_r.get("earnings_date")
        sig  = sd.get("signal", {})
        if ed and "BUY" in sig.get("signal",""):
            try:
                days_to = (datetime.date.fromisoformat(ed) - datetime.date.today()).days
                if 0 <= days_to <= 10:
                    rebal_actions.append({
                        "type": "WAIT", "ticker": ticker, "priority": 2,
                        "reason": "Earnings in {} days ({}). Wait for post-earnings reaction before adding. Risk of gap move both directions.".format(days_to, ed),
                        "color": "var(--warn)"
                    })
            except Exception: pass

    # 4. Buy signals — with ATR position sizing
    buy_candidates = []
    for ticker, sd in stocks_data.items():
        sig = sd.get("signal", {})
        if "BUY" in sig.get("signal", ""):
            fv  = sig.get("best_fv")
            pr  = sd.get("fund", {}).get("price")
            upside = (fv - pr) / pr * 100 if (fv and pr) else 0
            buy_candidates.append((upside, ticker, sd))

    buy_candidates.sort(reverse=True)
    for upside, ticker, sd in buy_candidates[:5]:
        sig  = sd.get("signal", {})
        fund = sd.get("fund", {})
        tech = sd.get("tech", {})
        pr   = fund.get("price") or 0
        atr_v = tech.get("atr")
        sizing = atr_position_size(total_val, atr_v, pr) if atr_v else {}
        trade_detail = ""
        if sizing:
            trade_detail = "  ATR sizing (1% risk): ~{} shares (${:,.0f}), stop ${:.2f}.".format(
                sizing["shares"], sizing["dollar_amount"], sizing["stop_price"])
        rebal_actions.append({
            "type": "BUY", "ticker": ticker, "priority": 3,
            "reason": "{:.0f}% upside to {} fair value. ".format(upside, sig.get("best_method","")) +
                      " ".join(sig.get("reasons",{}).get("buy",[])[:2]) + trade_detail,
            "color": "var(--up)"
        })

    # 5. Cash notes
    if metrics["cash_pct"] > 20:
        rebal_actions.append({
            "type": "DEPLOY", "ticker": "CASH", "priority": 2,
            "reason": "{:.1f}% cash drag — consider deploying into highest-conviction BUY candidates above.".format(metrics["cash_pct"]),
            "color": "var(--warn)"
        })
    elif metrics["cash_pct"] < 3:
        rebal_actions.append({
            "type": "BUILD CASH", "ticker": "CASH", "priority": 3,
            "reason": "Cash is only {:.1f}%. Trim overvalued/overweight positions to maintain a liquidity buffer.".format(metrics["cash_pct"]),
            "color": "var(--warn)"
        })

    rebal_actions.sort(key=lambda x: x["priority"])

    rebal_html = ""
    for action in rebal_actions:
        type_icons = {"TRIM":"✂", "SELL":"🔻", "BUY":"⬆", "DEPLOY":"💵", "BUILD CASH":"🏦", "WAIT":"⏳"}
        icon = type_icons.get(action["type"], "•")
        rebal_html += """
        <div class="rebal-card" style="border-left:4px solid {col};">
          <div class="rebal-action" style="color:{col};">{icon}</div>
          <div class="rebal-body">
            <div class="rebal-ticker" style="color:{col};">{action_type}: {tkr}</div>
            <div class="rebal-reason">{reason}</div>
          </div>
        </div>""".format(
            col=action["color"], icon=icon,
            action_type=action["type"], tkr=action["ticker"],
            reason=action["reason"]
        )

    if not rebal_html:
        rebal_html = '<div style="color:var(--up);padding:24px;background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.2);border-radius:10px;font-size:15px;">✓ Portfolio looks balanced. No urgent rebalancing actions identified.</div>'

    rebal_tab = """
<div id="tab-rebalance" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Rebalance Advisor</div>
  <div class="stat-grid" style="margin-bottom:32px;">
    <div class="stat-card">
      <div class="stat-label">Total Value</div>
      <div class="stat-value" style="font-size:28px;">{tv}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Cash Available</div>
      <div class="stat-value" style="font-size:28px;color:var(--warn);">{cash}</div>
      <div class="stat-sub">{cpct:.1f}% of portfolio</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Buy Opportunities</div>
      <div class="stat-value" style="font-size:28px;color:var(--up);">{buys}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Sell / Trim</div>
      <div class="stat-value" style="font-size:28px;color:var(--down);">{sells_r}</div>
    </div>
  </div>
  {actions}
</div>
</div>""".format(
        tv=mo(metrics["total_value"]),
        cash=mo(metrics["cash"]),
        cpct=metrics["cash_pct"],
        buys=len([a for a in rebal_actions if a["type"] == "BUY"]),
        sells_r=len([a for a in rebal_actions if a["type"] in ("SELL","TRIM")]),
        actions=rebal_html
    )

    # ── Forecast Tab ────────────────────────────────────────────────
    forecast_cards = ""
    for ticker in sorted(stocks_data.keys()):
        sd       = stocks_data[ticker]
        fc       = sd.get("forecast", {})
        fund     = sd.get("fund", {})
        sig      = sd.get("signal", {})
        cls      = sd.get("classification", {})
        if not fc:
            continue
        price    = fund.get("price") or 0
        base     = fc.get("blended_base")
        bull     = fc.get("blended_bull")
        bear     = fc.get("blended_bear")
        conf     = fc.get("confidence", "LOW")
        exp_ret  = fc.get("expected_return_pct", 0)
        bull_ret = fc.get("bull_return_pct", 0)
        bear_ret = fc.get("bear_return_pct", 0)
        n_models = fc.get("n_models", 0)

        conf_color = {"HIGH": "var(--up)", "MEDIUM": "var(--warn)", "LOW": "var(--muted)"}.get(conf, "var(--muted)")
        ret_color  = "var(--up)" if exp_ret >= 0 else "var(--down)"
        t_col      = type_color(cls.get("type","BLEND"))

        # Model breakdown rows
        model_rows = ""
        for model_name, proj in fc.get("models", {}).items():
            w    = fc.get("model_weights", {}).get(model_name, 0)
            b    = proj.get("base")
            bull2 = proj.get("bull")
            bear2 = proj.get("bear")
            note  = proj.get("note","")
            if b and price > 0:
                ret_p = (b - price) / price * 100
                rc    = "var(--up)" if ret_p >= 0 else "var(--down)"
                model_rows += """<tr>
                  <td style="font-size:12px;color:var(--muted);padding:8px 0;">{mn}</td>
                  <td style="font-family:DM Mono,monospace;font-size:12px;text-align:right;">{base}</td>
                  <td style="font-family:DM Mono,monospace;font-size:12px;text-align:right;color:{rc};">{ret}</td>
                  <td style="font-family:DM Mono,monospace;font-size:11px;text-align:right;color:var(--up);">{bull2}</td>
                  <td style="font-family:DM Mono,monospace;font-size:11px;text-align:right;color:var(--down);">{bear2}</td>
                  <td style="font-size:10px;color:var(--muted);padding-left:12px;">{note}</td>
                  <td style="font-family:DM Mono,monospace;font-size:10px;text-align:right;color:var(--muted);">{w:.0%}</td>
                </tr>""".format(
                    mn=model_name, base=mo(b), rc=rc,
                    ret="{:+.0f}%".format(ret_p),
                    bull2=mo(bull2) if bull2 else "—",
                    bear2=mo(bear2) if bear2 else "—",
                    note=note[:55], w=w)

        # Range bar (bear → base → bull)
        if bear and bull and base and price > 0:
            all_vals = [bear, price, base, bull]
            mn_v = min(all_vals) * 0.95
            mx_v = max(all_vals) * 1.05
            rng  = mx_v - mn_v or 1
            def to_pct(v): return (v - mn_v) / rng * 100
            bear_x  = to_pct(bear)
            base_x  = to_pct(base)
            bull_x  = to_pct(bull)
            price_x = to_pct(price)
            range_bar = """
            <div style="position:relative;height:32px;background:var(--surface2);border-radius:6px;overflow:visible;margin:16px 0;">
              <!-- bear-to-bull fill -->
              <div style="position:absolute;left:{bl:.1f}%;width:{bw:.1f}%;height:100%;background:linear-gradient(90deg,rgba(224,92,92,0.25),rgba(0,200,150,0.25));border-radius:4px;"></div>
              <!-- base marker -->
              <div style="position:absolute;left:{bax:.1f}%;top:-4px;width:3px;height:40px;background:var(--accent);border-radius:2px;"></div>
              <!-- price marker -->
              <div style="position:absolute;left:{px:.1f}%;top:-4px;width:2px;height:40px;background:var(--muted);border-radius:2px;"></div>
              <!-- labels -->
              <div style="position:absolute;left:{bl:.1f}%;top:36px;font-size:9px;font-family:DM Mono,monospace;color:var(--down);white-space:nowrap;">{bear}</div>
              <div style="position:absolute;left:{bax:.1f}%;top:36px;font-size:9px;font-family:DM Mono,monospace;color:var(--accent);white-space:nowrap;">{base} ●</div>
              <div style="position:absolute;right:{buw:.1f}%;top:36px;font-size:9px;font-family:DM Mono,monospace;color:var(--up);white-space:nowrap;">{bull}</div>
              <div style="position:absolute;left:{px:.1f}%;top:36px;font-size:9px;font-family:DM Mono,monospace;color:var(--muted);white-space:nowrap;transform:translateY(12px);">now {pr}</div>
            </div>""".format(
                bl=bear_x, bw=bull_x-bear_x, bax=base_x, px=price_x,
                buw=100-bull_x,
                bear=mo(bear), base=mo(base), bull=mo(bull), pr=mo(price))
        else:
            range_bar = ""

        forecast_cards += """
        <div class="stock-card" style="margin-bottom:24px;">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
              <div style="margin-top:6px;">
                <span class="type-badge" style="color:{tcol};border-color:{tcol};">{ctype}</span>
              </div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:11px;color:var(--muted);">Current Price</div>
              <div style="font-family:DM Mono,monospace;font-size:18px;font-weight:700;">{price_d}</div>
            </div>
            <div style="margin-left:32px;text-align:center;min-width:120px;">
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);">12-Mo Base Target</div>
              <div style="font-family:DM Serif Display,serif;font-size:32px;color:{retcol};font-weight:700;">{base_d}</div>
              <div style="font-size:13px;color:{retcol};">{ret_d}</div>
            </div>
            <div style="margin-left:32px;text-align:center;">
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);">Confidence</div>
              <div style="font-size:18px;font-weight:700;color:{confcol};">{conf}</div>
              <div style="font-size:11px;color:var(--muted);">{n_models} models</div>
            </div>
          </div>
          <div class="stock-card-body">
            {range_bar}
            <div style="margin-top:48px;display:grid;grid-template-columns:1fr 1fr;gap:24px;">
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Bear / Base / Bull Scenarios</div>
                <div style="display:flex;gap:16px;margin-bottom:16px;">
                  <div style="flex:1;background:rgba(224,92,92,0.08);border:1px solid rgba(224,92,92,0.2);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:10px;color:var(--down);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Bear</div>
                    <div style="font-family:DM Mono,monospace;font-size:16px;font-weight:700;color:var(--down);">{bear_d}</div>
                    <div style="font-size:11px;color:var(--down);">{bear_ret}</div>
                  </div>
                  <div style="flex:1;background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:10px;color:var(--accent);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Base</div>
                    <div style="font-family:DM Mono,monospace;font-size:16px;font-weight:700;color:var(--accent);">{base_d}</div>
                    <div style="font-size:11px;color:var(--accent);">{exp_ret}</div>
                  </div>
                  <div style="flex:1;background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.2);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:10px;color:var(--up);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Bull</div>
                    <div style="font-family:DM Mono,monospace;font-size:16px;font-weight:700;color:var(--up);">{bull_d}</div>
                    <div style="font-size:11px;color:var(--up);">{bull_ret}</div>
                  </div>
                </div>
              </div>
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Model Breakdown</div>
                <table style="width:100%;border-collapse:collapse;">
                  <thead><tr>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:left;padding:4px 0;letter-spacing:1px;">Model</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Base</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Return</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Bull</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Bear</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:left;padding-left:12px;">Methodology</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Wgt</th>
                  </tr></thead>
                  <tbody>{model_rows}</tbody>
                </table>
              </div>
            </div>
          </div>
        </div>""".format(
            tkr=ticker,
            name=fund.get("company_name", ticker),
            tcol=t_col, ctype=cls.get("type","—"),
            price_d=mo(price),
            base_d=mo(base) if base else "—",
            bear_d=mo(bear) if bear else "—",
            bull_d=mo(bull) if bull else "—",
            ret_d="{:+.0f}%".format(exp_ret),
            retcol=ret_color,
            exp_ret="{:+.0f}%".format(exp_ret),
            bear_ret="{:+.0f}%".format(bear_ret),
            bull_ret="{:+.0f}%".format(bull_ret),
            conf=conf, confcol=conf_color,
            n_models=n_models,
            range_bar=range_bar,
            model_rows=model_rows,
        )

    forecast_tab = """
<div id="tab-forecast" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">12-Month Price Projection — 5 Independent Models</div>
  <div class="warn-banner" style="margin-bottom:24px;">
    <strong>How forecasts are built:</strong>
    Blended from 5 independent models — Analyst Consensus, NTM Earnings × Market Multiple,
    Revenue Momentum (EV/NTM Rev), Technical Momentum (trend extrapolation), and DCF Fundamental.
    Growth stocks weight analyst/earnings/revenue 30/25/20. Value stocks weight DCF/earnings/analyst 30/25/20.
    <strong>Confidence</strong> reflects how tightly the 5 models agree (CV&lt;15%=HIGH, &lt;30%=MEDIUM, else LOW).
    Bear scenario assumes multiple compression + trend reversal. Bull assumes expansion + acceleration.
    Not financial advice — projections are probabilistic, not guaranteed.
  </div>
  {cards}
</div>
</div>""".format(cards=forecast_cards or '<p style="color:var(--muted);">No forecast data available.</p>')

    # ── Sentiment Tab ───────────────────────────────────────────────
    sentiment_cards  = ""
    all_insider_buys = []   # [{ticker, company, name, role, shares, value, date}]
    for ticker in sorted(stocks_data.keys()):
        sd    = stocks_data[ticker]
        sent  = sd.get("sentiment", {})
        fund  = sd.get("fund", {})
        an    = sd.get("analyst", {})
        sig   = sd.get("signal", {})
        if not sent:
            continue

        s_score  = sent.get("sentiment_score", 0)
        s_label  = sent.get("sentiment_label", "Neutral")
        s_col    = {"Bullish": "var(--up)", "Slightly Bullish": "var(--up)",
                    "Bearish": "var(--down)", "Slightly Bearish": "var(--down)"}.get(s_label, "var(--warn)")

        inst_own  = sent.get("inst_own_pct")
        inst_net  = sent.get("inst_net_flow")
        inst_buy  = sent.get("inst_buyers_count")
        inst_sell = sent.get("inst_sellers_count")

        ins_buy  = sent.get("insider_buy_count")
        ins_sell = sent.get("insider_sell_count")
        ins_net  = sent.get("insider_net")

        # Institutional holders table
        top_holders_rows = ""
        for h in (sent.get("inst_top_holders") or [])[:8]:
            chg = h.get("change")
            chg_disp = "{:+,}".format(chg) if chg else "—"
            chg_col  = "var(--up)" if (chg and chg > 0) else "var(--down)" if (chg and chg < 0) else "var(--muted)"
            top_holders_rows += """<tr>
              <td style="font-size:12px;padding:8px 0;">{nm}</td>
              <td style="font-family:DM Mono,monospace;font-size:12px;text-align:right;">{pct:.2f}%</td>
              <td style="font-family:DM Mono,monospace;font-size:12px;text-align:right;color:{cc};">{chg}</td>
            </tr>""".format(nm=h["name"][:30], pct=h["pct"], cc=chg_col, chg=chg_disp)

        # Insider transactions table
        insider_rows = ""
        for tx in (sent.get("insider_transactions") or [])[:8]:
            is_b = tx.get("is_buy") or tx.get("type") == "BUY"
            is_s = tx.get("type") == "SELL"
            tc   = "var(--up)" if is_b else "var(--down)" if is_s else "var(--muted)"
            bg   = "background:rgba(0,200,150,0.07);" if is_b else ""
            sh   = "{:,}".format(tx["shares"]) if tx.get("shares") else "—"
            vl   = mo(tx["value"]) if tx.get("value") else "—"
            dt   = tx.get("date", "")[:7] or "—"   # show YYYY-MM
            tp   = tx.get("type") or "—"
            insider_rows += """<tr style="{bg}">
              <td style="font-size:12px;padding:6px 0;">{nm}</td>
              <td style="font-size:11px;color:var(--muted);">{role}</td>
              <td style="font-family:DM Mono,monospace;font-size:11px;color:{tc};font-weight:700;">{tp}</td>
              <td style="font-family:DM Mono,monospace;font-size:11px;text-align:right;">{sh}</td>
              <td style="font-family:DM Mono,monospace;font-size:11px;text-align:right;">{vl}</td>
              <td style="font-family:DM Mono,monospace;font-size:10px;color:var(--muted);text-align:right;">{dt}</td>
            </tr>""".format(nm=tx["name"][:22], role=tx["role"][:18], tc=tc, bg=bg, tp=tp, sh=sh, vl=vl, dt=dt)

        # Analyst target range visual
        t_mean   = an.get("target_price")
        t_high   = an.get("target_high")
        t_low    = an.get("target_low")
        t_med    = an.get("target_median") or an.get("target_price")
        price    = fund.get("price") or 0
        n_an     = an.get("num_analysts") or "?"

        target_range_html = ""
        if t_mean and t_low and t_high and price > 0:
            mn_v = min(t_low, price) * 0.97
            mx_v = max(t_high, price) * 1.03
            rng  = mx_v - mn_v or 1
            def tp(v): return (v - mn_v) / rng * 100
            target_range_html = """
            <div style="background:var(--surface2);border-radius:8px;padding:16px;margin-bottom:16px;">
              <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:16px;">Analyst Price Target Range ({n_an} analysts)</div>
              <div style="position:relative;height:28px;background:var(--surface);border-radius:4px;overflow:visible;">
                <div style="position:absolute;left:{lo:.1f}%;width:{rw:.1f}%;height:100%;background:rgba(79,142,247,0.2);border-radius:4px;"></div>
                <div style="position:absolute;left:{mn_x:.1f}%;top:-3px;width:2px;height:34px;background:var(--accent);"></div>
                <div style="position:absolute;left:{pr_x:.1f}%;top:-3px;width:2px;height:34px;background:var(--muted);"></div>
                <div style="position:absolute;left:{lo:.1f}%;top:32px;font-size:10px;font-family:DM Mono,monospace;color:var(--muted);">Low {tl}</div>
                <div style="position:absolute;left:{mn_x:.1f}%;top:32px;font-size:10px;font-family:DM Mono,monospace;color:var(--accent);">Mean {tm}</div>
                <div style="position:absolute;right:{hi_r:.1f}%;top:32px;font-size:10px;font-family:DM Mono,monospace;color:var(--muted);">High {th}</div>
                <div style="position:absolute;left:{pr_x:.1f}%;top:46px;font-size:10px;font-family:DM Mono,monospace;color:var(--muted);">Now {pr}</div>
              </div>
            </div>""".format(
                n_an=n_an,
                lo=tp(t_low), rw=tp(t_high)-tp(t_low),
                mn_x=tp(t_mean), pr_x=tp(price),
                hi_r=100-tp(t_high),
                tl=mo(t_low), tm=mo(t_mean), th=mo(t_high), pr=mo(price))

        sentiment_cards += """
        <div class="stock-card" style="margin-bottom:24px;">
          <div class="stock-card-header">
            <div>
              <div class="sc-ticker">{tkr}</div>
              <div class="sc-name">{name}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:11px;color:var(--muted);">Sentiment</div>
              <div style="font-family:DM Serif Display,serif;font-size:24px;font-weight:700;color:{scol};">{slbl}</div>
              <div style="font-size:11px;color:var(--muted);">Score {ss:+d} / 3</div>
            </div>
          </div>
          <div class="stock-card-body">
            {tgt_range}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
              <div>
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Institutional Ownership</div>
                <div class="fund-block">
                  <div class="fund-row"><span class="fund-key">Total Inst. Ownership</span><span class="fund-val">{inst_own}</span></div>
                  <div class="fund-row"><span class="fund-key">Institutions Buying</span><span class="fund-val" style="color:var(--up);">{inst_buy}</span></div>
                  <div class="fund-row"><span class="fund-key">Institutions Selling</span><span class="fund-val" style="color:var(--down);">{inst_sell}</span></div>
                  <div class="fund-row"><span class="fund-key">Net Flow (Buy − Sell)</span><span class="fund-val" style="color:{inst_net_col};">{inst_net_disp}</span></div>
                </div>
                {inst_tbl}
                <div style="margin-top:16px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Insider Activity (6mo)</div>
                <div class="fund-block">
                  <div class="fund-row"><span class="fund-key">Insider Purchases</span><span class="fund-val" style="color:var(--up);">{ins_buy}</span></div>
                  <div class="fund-row"><span class="fund-key">Insider Sales</span><span class="fund-val" style="color:var(--down);">{ins_sell}</span></div>
                  <div class="fund-row"><span class="fund-key">Net Insider Flow</span><span class="fund-val" style="color:{ins_net_col};">{ins_net_disp}</span></div>
                </div>
              </div>
              <div>
                {insider_tbl}
                <div style="margin-top:16px;">
                  <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;">Sentiment Signals</div>
                  <ul class="reason-list">
                    {s_reasons}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>""".format(
            tkr=ticker,
            name=fund.get("company_name", ticker),
            scol=s_col, slbl=s_label,
            ss=s_score,
            tgt_range=target_range_html,
            inst_own="{:.1f}%".format(inst_own) if inst_own else "—",
            inst_buy=str(inst_buy) if inst_buy is not None else "—",
            inst_sell=str(inst_sell) if inst_sell is not None else "—",
            inst_net_col="var(--up)" if (inst_net and inst_net > 0) else "var(--down)" if inst_net else "var(--muted)",
            inst_net_disp=("{:+d}".format(inst_net) if inst_net is not None else "—"),
            inst_tbl=("""
                <div style="margin-top:12px;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;">Top Institutional Holders</div>
                <table style="width:100%;border-collapse:collapse;">
                  <thead><tr>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:left;padding:4px 0;">Institution</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">% Held</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Chg Shares</th>
                  </tr></thead>
                  <tbody>{th_rows}</tbody>
                </table>""".format(th_rows=top_holders_rows) if top_holders_rows else ""),
            ins_buy=str(ins_buy) if ins_buy is not None else "—",
            ins_sell=str(ins_sell) if ins_sell is not None else "—",
            ins_net_col="var(--up)" if (ins_net and ins_net > 0) else "var(--down)" if (ins_net and ins_net < 0) else "var(--muted)",
            ins_net_disp=("{:+d}".format(ins_net) if ins_net is not None else "—"),
            insider_tbl=("""
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;">Recent Insider Transactions</div>
                <table style="width:100%;border-collapse:collapse;">
                  <thead><tr>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:left;padding:4px 0;">Insider</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:left;">Role</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);">Type</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Shares</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Value</th>
                    <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);text-align:right;">Date</th>
                  </tr></thead>
                  <tbody>{ir}</tbody>
                </table>""".format(ir=insider_rows) if insider_rows else ""),
            s_reasons="".join("<li>{}</li>".format(r) for r in sent.get("sentiment_reasons",[])) or "<li style='color:var(--muted);'>No significant institutional/insider signals detected.</li>",
        )

        # Collect insider buys for portfolio-wide summary
        for tx in (sent.get("insider_transactions") or []):
            if tx.get("is_buy") or tx.get("type") == "BUY":
                all_insider_buys.append({
                    "ticker":  ticker,
                    "company": fund.get("company_name", ticker),
                    "name":    tx.get("name", ""),
                    "role":    tx.get("role", ""),
                    "shares":  tx.get("shares"),
                    "value":   tx.get("value"),
                    "date":    tx.get("date", ""),
                })

    # ── Build insider buying summary banner ─────────────────────────
    insider_buy_banner = ""
    if all_insider_buys:
        # Sort by value descending (None → end)
        all_insider_buys.sort(key=lambda x: x["value"] or 0, reverse=True)
        n_tickers = len({b["ticker"] for b in all_insider_buys})
        n_buys    = len(all_insider_buys)
        buy_rows  = ""
        for b in all_insider_buys[:20]:
            sh  = "{:,}".format(b["shares"]) if b.get("shares") else "—"
            vl  = mo(b["value"]) if b.get("value") else "—"
            dt  = b.get("date", "")[:7] or "—"
            buy_rows += (
                "<tr>"
                "<td style='font-family:DM Mono,monospace;font-size:13px;font-weight:700;"
                "color:var(--up);padding:8px 12px 8px 0;'>{tkr}</td>"
                "<td style='font-size:12px;color:var(--text);padding:8px 8px 8px 0;'>{nm}</td>"
                "<td style='font-size:11px;color:var(--muted);padding:8px 8px 8px 0;'>{role}</td>"
                "<td style='font-family:DM Mono,monospace;font-size:12px;text-align:right;"
                "padding:8px 8px 8px 0;'>{sh}</td>"
                "<td style='font-family:DM Mono,monospace;font-size:12px;text-align:right;"
                "color:var(--up);font-weight:600;padding:8px 0;'>{vl}</td>"
                "<td style='font-family:DM Mono,monospace;font-size:11px;color:var(--muted);"
                "text-align:right;padding:8px 0;'>{dt}</td>"
                "</tr>"
            ).format(tkr=b["ticker"], nm=b["name"][:26], role=b["role"][:20],
                     sh=sh, vl=vl, dt=dt)
        insider_buy_banner = """
  <div style="background:rgba(0,200,150,0.07);border:1px solid rgba(0,200,150,0.3);
              border-radius:10px;padding:20px 24px;margin-bottom:28px;">
    <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                color:var(--up);margin-bottom:14px;font-weight:700;">
      Insider Buying — {nb} purchase{ps} across {nt} holding{hs} (last 6 months)
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:left;padding:4px 12px 4px 0;">Ticker</th>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:left;padding:4px 8px 4px 0;">Insider</th>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:left;padding:4px 8px 4px 0;">Role</th>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:right;padding:4px 8px 4px 0;">Shares</th>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:right;padding:4px 8px 4px 0;">Value</th>
        <th style="font-size:9px;font-family:DM Mono,monospace;color:var(--muted);
                   text-align:right;padding:4px 0;">Date</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>""".format(nb=n_buys, ps="s" if n_buys != 1 else "",
                   nt=n_tickers, hs="s" if n_tickers != 1 else "",
                   rows=buy_rows)

    sentiment_tab = """
<div id="tab-sentiment" class="tab-panel">
<div class="container" style="padding-top:48px;">
  <div class="section-label">Market Sentiment — Institutional & Insider Activity</div>
  <div class="warn-banner" style="margin-bottom:24px;">
    <strong>Data sources:</strong>
    Institutional ownership from SEC 13F filings (via yfinance). Insider transactions from SEC Form 4 (last 6 months).
    Analyst price target range from consensus data.
    <strong>Interpretation:</strong> Net institutional buying = more institutions increased/opened than decreased/closed positions in the most recent 13F filing period.
    Insider sales are normal and often planned; <em>insider buying</em> (with own money, on the open market) is the stronger signal.
  </div>
  {insider_buy_banner}
  {cards}
</div>
</div>""".format(
        insider_buy_banner=insider_buy_banner,
        cards=sentiment_cards or '<p style="color:var(--muted);">No sentiment data available.</p>'
    )

    # ── Risk Tab ───────────────────────────────────────────────────
    risk_tab = _build_risk_tab_html(stocks_data, metrics)

    # ── Assemble full HTML ─────────────────────────────────────────
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Analysis Report</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('signals',this)">Signals</button>
  <button class="tab-btn" onclick="switchTab('forecast',this)">Forecast</button>
  <button class="tab-btn" onclick="switchTab('sentiment',this)">Sentiment</button>
  <button class="tab-btn" onclick="switchTab('technical',this)">Technical</button>
  <button class="tab-btn" onclick="switchTab('fundamentals',this)">Fundamentals</button>
  <button class="tab-btn" onclick="switchTab('earnings',this)">Earnings Calendar</button>
  <button class="tab-btn" onclick="switchTab('backtest',this)">Backtesting</button>
  <button class="tab-btn" onclick="switchTab('rebalance',this)">Rebalance</button>
  <button class="tab-btn" onclick="switchTab('risk',this)">Risk</button>
</div>

{overview}
{signals}
{forecast}
{sentiment}
{technical}
{fundamentals}
{earnings}
{backtest}
{rebalance}
{risk}

<div class="footer">
  Portfolio Analysis Report · Generated {ts}<br>
  Data: TradingView Screener (fundamentals) · yfinance (prices/technicals) · SEC EDGAR (historical)<br>
  <span style="color:#3a3f55;">This report is for informational purposes only. Not financial advice.</span>
</div>

<script>{js}</script>
</body>
</html>""".format(
        css=CSS, js=JS, ts=ts,
        overview=overview_tab,
        signals=signals_tab,
        forecast=forecast_tab,
        sentiment=sentiment_tab,
        technical=tech_tab,
        fundamentals=fund_tab,
        earnings=earnings_tab,
        backtest=bt_tab,
        rebalance=rebal_tab,
        risk=risk_tab,
    )
    return html


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Analysis Report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("portfolio", help="Path to portfolio CSV (ticker,shares columns)")
    parser.add_argument("--backtest", type=int, default=DEFAULT_BACKTEST_DAYS, metavar="DAYS",
        help="Days of price history to use for backtesting (default: {})".format(DEFAULT_BACKTEST_DAYS))
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PORTFOLIO ANALYSIS REPORT")
    print("=" * 60)

    # 1. Load CSV
    print("\n[1/5] Loading portfolio from {}...".format(args.portfolio))
    raw = load_portfolio(args.portfolio)
    cash_entry = raw.pop("CASH", {"shares": 0.0, "cost_basis": None})
    cash = cash_entry["shares"] if isinstance(cash_entry, dict) else float(cash_entry)
    tickers = [t for t in raw.keys()]
    print("  {} positions + ${:,.0f} cash".format(len(tickers), cash))

    # 2. Fetch data for each ticker (parallel — one thread per ticker)
    print("\n[2/5] Fetching fundamentals & price history (parallel)...")
    backtest_days = args.backtest

    def _fetch_ticker(ticker):
        entry      = raw[ticker]
        shares     = entry["shares"]     if isinstance(entry, dict) else float(entry)
        cost_basis = entry["cost_basis"] if isinstance(entry, dict) else None

        fund = fetch_tv_fundamentals(ticker)
        if not fund:
            return ticker, None, shares, cost_basis, [], {}, {}

        wacc_raw = fetch_wacc_inputs(ticker)
        fund["wacc_raw"] = wacc_raw

        bars      = fetch_price_history(ticker, backtest_days + 50)
        analyst   = fetch_analyst_forecasts(ticker)
        sentiment = fetch_market_sentiment(ticker)

        return ticker, fund, shares, cost_basis, bars, analyst, sentiment

    stocks_data = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as pool:
        futures = {pool.submit(_fetch_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                ticker, fund, shares, cost_basis, bars, analyst, sentiment = future.result()
            except Exception as exc:
                print("  {} ERROR: {}".format(futures[future], exc))
                continue

            if not fund:
                print("  {} SKIPPED (no data)".format(ticker))
                continue

            price = fund.get("price", 0) or 0
            value = price * shares

            # Cache analyst target in fund for Fwd PEG cap
            fund["_analyst_cache"] = analyst

            # Merge sentiment into analyst dict for signal access
            analyst["_sentiment"]     = sentiment
            analyst["target_median"]  = sentiment.get("target_median") or analyst.get("target_median")
            analyst["target_high"]    = sentiment.get("target_high")
            analyst["target_low"]     = sentiment.get("target_low")

            # Merge 52-week data from yfinance info into fund (supplement TV data)
            if analyst.get("week52_high"): fund["week52_high_yf"] = analyst["week52_high"]
            if analyst.get("week52_low"):  fund["week52_low_yf"]  = analyst["week52_low"]

            cb_str = "(cost ${:.2f})".format(cost_basis) if cost_basis else ""
            print("  {} ${:.2f}  {:.0f} bars  rec={}  target={}  inst_net={}  {}".format(
                ticker, price, len(bars),
                analyst.get("recommendation") or "—",
                "${:.0f}".format(analyst["target_price"]) if analyst.get("target_price") else "—",
                sentiment.get("inst_net_flow", "—"),
                cb_str).rstrip())

            stocks_data[ticker] = {
                "shares":     shares,
                "cost_basis": cost_basis,
                "value":      value,
                "fund":       fund,
                "bars":       bars,
                "analyst":    analyst,
                "sentiment":  sentiment,
            }

    # Restore original ticker order (as_completed is non-deterministic)
    stocks_data = {t: stocks_data[t] for t in tickers if t in stocks_data}

    # 3. Compute technicals, classification, valuation, backtest
    print("\n[3/5] Computing technicals & signals...")

    # Pre-fetch benchmarks once per unique (sector, industry) pair — avoids a redundant
    # TradingView query for every stock when multiple holdings share the same sector.
    _bm_cache = {}
    for _sd in stocks_data.values():
        _key = (_sd["fund"].get("sector"), _sd["fund"].get("industry"))
        if _key not in _bm_cache:
            _bm_cache[_key] = fetch_market_benchmarks(
                sector=_key[0], industry=_key[1])

    for ticker, sd in stocks_data.items():
        fund    = sd["fund"]
        bars    = sd["bars"]
        analyst = sd.get("analyst", {})

        tech = compute_technicals(bars) if bars else {}
        cls  = classify_stock(fund)

        benchmarks = _bm_cache.get(
            (fund.get("sector"), fund.get("industry")),
            fetch_market_benchmarks(sector=fund.get("sector"), industry=fund.get("industry")))

        bt        = backtest_methods(fund, bars, benchmarks) if bars else {}
        signal    = generate_signal(fund, tech, cls, bt, analyst, benchmarks)
        forecast  = project_price(fund, tech, analyst, benchmarks)
        sentiment = sd.get("sentiment", {})

        sd["tech"]           = tech
        sd["classification"] = cls
        sd["benchmarks"]     = benchmarks
        sd["backtest"]       = bt
        sd["signal"]         = signal
        sd["forecast"]       = forecast
        sd["sentiment"]      = sentiment

        print("  {}  {}  {}  [{} peers]  best={}".format(
            ticker, cls["type"], signal["signal"],
            benchmarks.get("peer_count", 0),
            bt.get("best_method", "—")))

    # 4. Portfolio metrics
    print("\n[4/5] Computing portfolio metrics...")
    metrics = compute_portfolio_metrics(stocks_data, cash)
    print("  Total value: ${:,.0f}  Cash: {:.1f}%  Positions: {}".format(
        metrics["total_value"], metrics["cash_pct"], metrics["n_stocks"]))
    if metrics["concentration_warns"]:
        print("  ⚠ Concentration warnings: " + ", ".join(metrics["concentration_warns"]))

    # 5. Generate report
    print("\n[5/5] Generating HTML report...")
    ts  = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")
    html = build_html(raw, stocks_data, metrics, ts)

    outdir = "portfolioData"
    os.makedirs(outdir, exist_ok=True)
    date_prefix = datetime.datetime.now().strftime("%Y_%m_%d")
    outfile = os.path.join(outdir, date_prefix + "_portfolio.html")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    print("  Saved: " + outfile)

    # Persist portfolio signals to ~/.valuation_suite/data.db (silent)
    _save_portfolio_signals(args.portfolio, stocks_data)

    print("\nOpening report in browser...")
    webbrowser.open("file://" + os.path.abspath(outfile))
    print("\nDone.")


def _save_portfolio_signals(portfolio_file: str, stocks_data: dict):
    """
    Save per-ticker signals from this portfolio run to ~/.valuation_suite/data.db.
    Silent try/except — never crashes the parent script.
    """
    try:
        import sqlite3 as _sq
        _db_dir  = os.path.join(os.path.expanduser("~"), ".valuation_suite")
        _db_path = os.path.join(_db_dir, "data.db")
        os.makedirs(_db_dir, exist_ok=True)
        _c = _sq.connect(_db_path)
        _c.execute(
            """CREATE TABLE IF NOT EXISTS portfolio_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date  TEXT NOT NULL,
                portfolio_file TEXT,
                ticker         TEXT NOT NULL,
                signal         TEXT,
                fair_value     REAL,
                upside_pct     REAL
            )"""
        )
        today  = datetime.datetime.now().strftime("%Y-%m-%d")
        pf_base = os.path.basename(portfolio_file) if portfolio_file else ""
        for ticker, sd in stocks_data.items():
            sig   = sd.get("signal", {})
            fund  = sd.get("fund", {})
            price = fund.get("price") or 0
            fv    = sig.get("best_fv")
            upside = ((fv - price) / price * 100) if (fv and price > 0) else None
            _c.execute(
                """INSERT INTO portfolio_signals
                   (snapshot_date, portfolio_file, ticker, signal, fair_value, upside_pct)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (today, pf_base, ticker, sig.get("signal"), fv, upside),
            )
        _c.commit()
        _c.close()
        print("  Portfolio signals saved to DB ({} positions).".format(len(stocks_data)))
    except Exception:
        pass   # never block the main report


if __name__ == "__main__":
    main()
