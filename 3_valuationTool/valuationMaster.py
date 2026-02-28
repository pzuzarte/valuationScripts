"""
Master Valuation Script — v7.2 (Backtest Logic Fix)
=====================================================
Generates a four-tab HTML valuation report for any stock on TradingView.
Data is fetched live from the TradingView screener API.

SETUP
-----
    pip install tradingview-screener

USAGE
-----
    python valuationMaster.py [TICKER] [--wacc RATE] [--backtest DAYS]

ARGUMENTS
---------
    TICKER          Stock ticker symbol. Defaults to AAPL if omitted.
                    Examples: NVDA, SHOP, MSFT, CRWD

    --wacc RATE     Override the auto-calculated WACC used in all DCF methods.
                    Provide as a decimal (e.g. --wacc 0.09 = 9%).
                    Default: CAPM formula (Rf + beta × ERP), typically 10–11%
                    for large-caps — which tends to be conservative vs analyst
                    consensus of 8–10%. Use this to align with external models.

    --backtest DAYS Number of trading days for the historical backtest (default: 1000 ≈ 4 years).
                    Ranks each valuation method by how closely its fair value tracked
                    the actual stock price over that window. Adds a 'Backtesting' tab
                    with a ranked table, chart, and verdict.
                    Set to 0 to skip the backtest entirely.
                    Example: --backtest 252  (≈ 1 full trading year)
                    Example: --backtest 500  (≈ 2 years)
                    Example: --backtest 0    (skip backtest, faster run)

EXAMPLES
--------
    python valuationMaster.py                           # AAPL, 1000-day backtest
    python valuationMaster.py NVDA                      # Nvidia, 1000-day backtest
    python valuationMaster.py AAPL --wacc 0.09          # AAPL, 9% WACC override
    python valuationMaster.py SHOP --backtest 252       # Shopify, 1-year backtest
    python valuationMaster.py MSFT --backtest 0         # MSFT, skip backtest (faster)

OUTPUT
------
    valuationData/YYYY_MM_DD_valuation_TICKER.html — opens automatically in your browser.

CLASSIC TAB    DCF · P/FCF · P/E (TTM) · EV/EBITDA · convergence analysis
GROWTH TAB     Reverse DCF · Forward PEG · EV/NTM Revenue · TAM Scenario · Rule of 40
BACKTEST TAB   Historical price tracking accuracy ranked across all methods (requires --backtest)
GLOSSARY TAB   Definitions of all terms and acronyms used in the report

CHANGES IN v7.2
--------------
    - FIXED: Backtest verdict text now compares fair value on day-1 against
      the price on day-1 (not today's price). Previously it was comparing a
      2020 estimate against a 2026 price, making every method look inaccurate.
    - FIXED: "Directional signal accuracy" now measured over the first 90
      trading days only, not the full backtest window. A BUY/SELL signal from
      a static model cannot be expected to predict a multi-year price path.
    - NEW: "Point-in-time error" column added to the table — shows how far each
      model's day-1 fair value was from the actual price on that same day,
      separately from MAPE (which measures tracking across the full window).
    - FIXED: Table column headers and footer now correctly describe what each
      metric means (MAPE vs point-in-time error vs directional accuracy).
    - FIXED: Limitations footnote updated — no longer references Yahoo Finance
      (data comes from SEC EDGAR + Stooq) and no longer incorrectly says this
      is a "single-period backtest" (it is a full time-series MAPE).
    - FIXED: Methodology note updated to clearly distinguish tracking accuracy,
      point-in-time accuracy, and directional signal accuracy.

CHANGES IN v7
-------------
    - Fixed backtest chart: solid lines now show historical fair values, dashed
      lines show the CURRENT prediction (what each model says the stock is worth
      NOW based on live TradingView data). Previously all non-best methods were
      dashed and no current prediction was drawn.
    - Added "Current Fair Value Predictions" card row below the chart showing
      each method's live prediction with upside/downside vs current price.
    - Prediction zone (rightmost 10% of chart) clearly delineated with a
      subtle background and "CURRENT" label.
    - Chart legend updated to explain solid vs dashed line meaning.
"""

import sys
import argparse
import datetime
import webbrowser
import os
import json
import urllib.request
import urllib.error
import urllib.parse
from tradingview_screener import Query, col
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valuation_models import (
    PROJECTION_YEARS, TERMINAL_GROWTH_RATE, MARGIN_OF_SAFETY,
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM, CONVERGENCE_THRESHOLD,
    calculate_wacc, growth_adjusted_multiples,
    run_dcf, run_pfcf, run_pe, run_ev_ebitda,
    calibrate_erg_multiple, run_reverse_dcf, run_forward_peg,
    run_ev_ntm_revenue, run_tam_scenario, run_rule_of_40,
    run_erg_valuation, run_graham_number, analyse_convergence, assess_reliability,
    run_monte_carlo_dcf, run_three_stage_dcf, run_rim,
    run_roic_excess_return, run_fcf_yield, run_ddm_hmodel,
    run_ncav, run_scurve_tam, run_pie, run_mean_reversion,
    run_bayesian_ensemble, run_multifactor_price_target,
    score_model_applicability,
)
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
#  CONFIRMED VALID TRADINGVIEW FIELDS
# ─────────────────────────────────────────────────────────────────

FIELDS = [
    "name", "description", "close", "market_cap_basic",
    "total_shares_outstanding", "float_shares_outstanding",
    "total_debt", "free_cash_flow", "beta_1_year",
    "earnings_per_share_diluted_ttm", "earnings_per_share_basic_ttm",
    "enterprise_value_ebitda_ttm", "gross_margin_percent_ttm",
    "net_income", "total_revenue", "operating_margin",
    "price_earnings_ttm", "debt_to_equity",
    "ebitda", "cash_per_share_fy",
    "sector",
    "industry",
]

# v4: fetched in a resilient second pass
FIELDS_V4 = [
    "earnings_per_share_forecast_next_fy",
    "revenue_forecast_next_fy",
    "total_revenue_yoy_growth_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm",
]


# ─────────────────────────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────────────────────────

def fetch_tv_data(ticker: str) -> dict:
    ticker_upper = ticker.upper()
    count, df = (
        Query().select(*FIELDS)
        .where(col("name") == ticker_upper)
        .limit(10).get_scanner_data()
    )
    if df.empty:
        count, df = (
            Query().select(*FIELDS)
            .where(col("market_cap_basic") > 0)
            .limit(3000).get_scanner_data()
        )
        df = df[df["ticker"].str.upper().str.endswith(":" + ticker_upper)]
    if df.empty:
        raise ValueError("Could not find '" + ticker + "' in TradingView screener.")

    row = df.iloc[0]

    def safe(field, default=None):
        val = row.get(field, default)
        if val is None or val != val:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    shares  = safe("total_shares_outstanding") or safe("float_shares_outstanding")
    price   = safe("close", 0)
    mktcap  = safe("market_cap_basic", 0)
    debt    = safe("total_debt", 0)
    fcf     = safe("free_cash_flow")
    rev     = safe("total_revenue")
    ni      = safe("net_income")
    op_mar  = safe("operating_margin")
    ev_eb   = safe("enterprise_value_ebitda_ttm")
    eps     = safe("earnings_per_share_diluted_ttm") or safe("earnings_per_share_basic_ttm")

    fcf_margin = (fcf / rev) if (fcf and rev and rev > 0) else None
    if fcf_margin and fcf_margin > 0.40:   growth = 0.15
    elif fcf_margin and fcf_margin > 0.20: growth = 0.12
    elif fcf_margin and fcf_margin > 0.10: growth = 0.09
    else:                                   growth = 0.06

    # ── EBITDA — from TradingView screener field ───────────────────────────
    ebitda = safe("ebitda")
    ebitda_method = "TradingView screener (ebitda)" if ebitda is not None else "N/A"

    # ── Cash — from TradingView screener field ─────────────────────────────
    cash_per_share = safe("cash_per_share_fy")
    cash_est = 0.0
    cash_source = "assumed zero (no data)"
    if cash_per_share is not None and shares:
        cash_est = cash_per_share * shares
        cash_source = "cash_per_share_fy × shares outstanding"
    cash_est = max(0.0, cash_est)

    ev_approx = mktcap + debt - cash_est   # corrected EV

    # ── v4: second-pass growth/forecast fields ─────────────────────────────
    fwd_eps=None; fwd_rev=None; rev_growth_pct=None; eps_growth_pct=None
    try:
        _, dfv4 = (Query().select(*FIELDS_V4)
            .where(col("name") == ticker_upper).limit(10).get_scanner_data())
        if not dfv4.empty:
            rv4 = dfv4.iloc[0]
            def safev4(f):
                v=rv4.get(f)
                if v is None or v!=v: return None
                try: return float(v)
                except: return None
            fwd_eps=safev4("earnings_per_share_forecast_next_fy")
            fwd_rev=safev4("revenue_forecast_next_fy")
            rev_growth_pct=safev4("total_revenue_yoy_growth_ttm")
            eps_growth_pct=safev4("earnings_per_share_diluted_yoy_growth_ttm")
    except Exception:
        pass

    print("  Forecast data: fwd_eps=" + str(fwd_eps) + "  fwd_rev=" + str(fwd_rev) +
          "  rev_growth_pct=" + str(rev_growth_pct) + "  eps_growth_pct=" + str(eps_growth_pct))

    # ── WACC inputs: fetch from yfinance ───────────────────────────────────
    wacc_raw = fetch_wacc_inputs(ticker_upper)

    # Growth rate priority:
    # 1. If analyst fwd EPS available → implied 1yr EPS growth (blended 70/30 with proxy)
    # 2. If TTM revenue growth available → use directly (regardless of fwd_eps)
    # 3. FCF-margin proxy (last resort)
    # Note: fwd_eps and fwd_rev are stored separately and used DIRECTLY in each
    # valuation method as the earnings/revenue anchor — the growth rate here only
    # governs multiples and DCF projection rate.
    growth_source = "FCF-margin proxy"
    if rev_growth_pct is not None and fwd_eps and eps and eps > 0 and fwd_eps != eps:
        # Both available: blend implied EPS growth with actual revenue growth
        eps_implied = (fwd_eps / eps) - 1.0
        rev_g_dec   = rev_growth_pct / 100.0
        growth      = max(0.02, min(eps_implied * 0.5 + rev_g_dec * 0.5, 0.80))
        growth_source = "analyst fwd EPS + TTM revenue (50/50 blend)"
    elif fwd_eps and eps and eps > 0 and fwd_eps > eps:
        eps_implied = (fwd_eps / eps) - 1.0
        growth      = max(0.02, min(eps_implied * 0.7 + growth * 0.3, 0.80))
        growth_source = "analyst fwd EPS (blended with FCF proxy)"
    elif rev_growth_pct is not None:
        growth = max(0.02, min(rev_growth_pct / 100.0, 0.80))
        growth_source = "TTM reported revenue growth"

    return {
        "ticker":        ticker_upper,
        "name":          str(row.get("name", ticker_upper)),
        "company_name":  str(row.get("description", "") or "").strip() or ticker_upper,
        "price":         price,
        "market_cap":    mktcap,
        "fcf":           fcf,
        "fcf_margin":    fcf_margin,
        "fcf_per_share": (fcf / shares) if (fcf and shares) else None,
        "cash":          cash_est,
        "cash_source":   cash_source,
        "total_debt":    debt,
        "ev_approx":     ev_approx,
        "ebitda":        ebitda,
        "ebitda_method": ebitda_method,
        "tv_ev_ebitda":  ev_eb,
        "eps":           eps,
        "fwd_eps":       fwd_eps,
        "fwd_rev":       fwd_rev,
        "rev_growth_pct":   rev_growth_pct,
        "eps_growth_pct":   eps_growth_pct,
        "growth_source":    growth_source,
        "current_pe":    (price / eps) if (eps and eps > 0) else None,
        "current_pfcf":  (mktcap / fcf) if (fcf and fcf > 0) else None,
        "current_ev_ebitda": (ev_approx / ebitda) if (ebitda and ebitda > 0) else None,
        "beta":          safe("beta_1_year", 1.0) or 1.0,
        "shares":        shares,
        "net_income":    ni,
        "revenue":       rev,
        "op_margin":     op_mar,
        "gross_margin":  safe("gross_margin_percent_ttm"),
        "est_growth":    growth,
        "pe":            safe("price_earnings_ttm"),
        "debt_equity":   safe("debt_to_equity"),
        "peg":           ((price / eps) / (growth * 100)) if (eps and eps > 0 and price) else None,
        "sector":        str(row.get("sector", "")) or None,
        "industry":      str(row.get("industry", "")) or None,
        "wacc_override":  None,   # filled in by main() if --wacc flag is used
        "wacc_raw":       wacc_raw,  # raw WACC components from third-pass fetch
    }


def fetch_market_benchmarks(sector: str = None, industry: str = None) -> dict:
    """
    Fetch live median P/E, P/FCF, and EV/EBITDA multiples.

    Uses a 3-tier hierarchy to find the tightest valid peer set:

        Tier 1 — Industry  (e.g. "Semiconductors")
            Most specific; same business model and competitive dynamics.
            Requires ≥ 10 peers with valid P/E.

        Tier 2 — Sector  (e.g. "Technology")
            Falls back here if industry is unknown or the peer set is too thin.

        Tier 3 — Broad Market (top-200 by market cap)
            Last resort when neither industry nor sector yields enough data.

    Returns dict with keys: pe, pfcf, ev_ebitda, sector_name, peer_count.
    """
    defaults = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0,
                "sector_name": "Broad Market", "peer_count": 0}

    BM_FIELDS = ["name", "market_cap_basic", "free_cash_flow",
                 "price_earnings_ttm", "enterprise_value_ebitda_ttm"]

    def _calc_medians(df):
        """Compute medians from a DataFrame; returns (benchmarks_dict, True) or (None, False)."""
        if df.empty or len(df) < 10:
            return None, False

        bm = {}
        pe_col = df["price_earnings_ttm"].dropna()
        pe_col = pe_col[(pe_col > 5) & (pe_col < 100)]
        if len(pe_col) < 8:
            return None, False
        bm["pe"] = round(float(pe_col.median()), 1)

        df = df.copy()
        df["pfcf"] = df["market_cap_basic"] / df["free_cash_flow"].replace(0, float('nan'))
        pf_col = df["pfcf"].dropna()
        pf_col = pf_col[(pf_col > 5) & (pf_col < 100)]
        bm["pfcf"] = round(float(pf_col.median()), 1) if len(pf_col) > 8 else bm["pe"]

        ev_col = df["enterprise_value_ebitda_ttm"].dropna()
        ev_col = ev_col[(ev_col > 3) & (ev_col < 60)]
        bm["ev_ebitda"] = round(float(ev_col.median()), 1) if len(ev_col) > 8 else 14.0

        return bm, True

    # ── Tier 1: industry peers ─────────────────────────────────────
    if industry:
        try:
            _, df_ind = (
                Query()
                .select(*BM_FIELDS)
                .where(col("industry") == industry,
                       col("market_cap_basic") > 500e6,
                       col("price_earnings_ttm") > 0)
                .order_by("market_cap_basic", ascending=False)
                .limit(500).get_scanner_data()
            )
            bm, ok = _calc_medians(df_ind)
            if ok:
                bm["sector_name"] = industry
                bm["peer_count"]  = len(df_ind)
                return bm
        except Exception:
            pass

    # ── Tier 2: sector peers ───────────────────────────────────────
    if sector:
        try:
            _, df_sec = (
                Query()
                .select(*BM_FIELDS)
                .where(col("sector") == sector,
                       col("market_cap_basic") > 1e9,
                       col("price_earnings_ttm") > 0)
                .order_by("market_cap_basic", ascending=False)
                .limit(500).get_scanner_data()
            )
            bm, ok = _calc_medians(df_sec)
            if ok:
                bm["sector_name"] = "{} (sector)".format(sector)
                bm["peer_count"]  = len(df_sec)
                return bm
        except Exception:
            pass

    # ── Tier 3: broad market ───────────────────────────────────────
    try:
        _, df_broad = (
            Query()
            .select(*BM_FIELDS)
            .where(col("market_cap_basic") > 5e9,
                   col("price_earnings_ttm") > 0)
            .order_by("market_cap_basic", ascending=False)
            .limit(200).get_scanner_data()
        )
        bm, ok = _calc_medians(df_broad)
        if ok:
            if industry:
                label = "Broad Market (industry '{}' too thin)".format(industry)
            elif sector:
                label = "Broad Market (sector fallback)"
            else:
                label = "Broad Market"
            bm["sector_name"] = label
            bm["peer_count"]  = len(df_broad)
            return bm
    except Exception:
        pass

    return defaults


def fetch_peer_erg_data(sector: str = None, industry: str = None) -> dict:
    """
    Fetch peers from TradingView and compute each peer's ERG multiple:

        peer_erg_i = (EV_i / Revenue_i) / rev_growth_pct_i

    This is the EV/Revenue-to-Growth ratio — the market price for one
    percentage point of revenue growth in this cohort right now.

    Uses the same 3-tier hierarchy as fetch_market_benchmarks:

        Tier 1 — Industry  (tightest, most meaningful peer set)
        Tier 2 — Sector    (fallback if industry is too thin)
        Tier 3 — Broad Market (last resort)

    Filtering rules:
      - Market cap  > $500M
      - Revenue growth  3% – 150%
      - EV/Revenue  0.5× – 60×
      - Peer ERG    0.02 – 3.0

    Returns a dict:
      {
        "multiples":   list of {"ticker", "erg", "ev_rev", "growth", "gm", "ro40"}
        "median":      float
        "p25":         float
        "p75":         float
        "p10":         float
        "p90":         float
        "peer_count":  int
        "sector_name": str   — describes which tier was used
        "fallback":    bool  — True if broad-market tier was used
      }
    """
    MIN_MKTCAP  = 500e6
    MIN_GROWTH  = 3.0
    MAX_GROWTH  = 150.0
    MIN_EV_REV  = 0.5
    MAX_EV_REV  = 60.0
    MIN_ERG     = 0.02
    MAX_ERG     = 3.0
    MIN_PEERS   = 8

    ERG_FIELDS = [
        "name", "market_cap_basic", "total_debt", "total_revenue",
        "total_revenue_yoy_growth_ttm", "gross_margin_percent_ttm",
        "free_cash_flow",
    ]

    DEFAULTS = {
        "multiples": [], "median": 0.50, "p25": 0.30, "p75": 0.70,
        "p10": 0.18, "p90": 1.10,
        "peer_count": 0, "sector_name": "Default (no data)", "fallback": True,
    }

    def _extract_rows(df):
        rows = []
        for _, row in df.iterrows():
            try:
                mktcap    = float(row.get("market_cap_basic") or 0)
                debt      = float(row.get("total_debt") or 0)
                rev       = float(row.get("total_revenue") or 0)
                rev_g_pct = float(row.get("total_revenue_yoy_growth_ttm") or 0)
                gm        = float(row.get("gross_margin_percent_ttm") or 0)
                fcf_val   = float(row.get("free_cash_flow") or 0)
                tkr       = str(row.get("name", ""))

                if mktcap < MIN_MKTCAP:                              continue
                if not (MIN_GROWTH <= rev_g_pct <= MAX_GROWTH):      continue
                if rev <= 0:                                          continue

                ev     = mktcap + max(0.0, debt)
                ev_rev = ev / rev
                if not (MIN_EV_REV <= ev_rev <= MAX_EV_REV):         continue

                erg = ev_rev / rev_g_pct
                if not (MIN_ERG <= erg <= MAX_ERG):                  continue

                fcf_margin_pct = (fcf_val / rev * 100) if rev > 0 else 0.0
                ro40 = rev_g_pct + fcf_margin_pct

                rows.append({
                    "ticker": tkr, "erg": round(erg, 4),
                    "ev_rev": round(ev_rev, 2), "growth": round(rev_g_pct, 2),
                    "gm": round(gm, 2), "ro40": round(ro40, 2),
                })
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return rows

    def _build_stats(rows, sector_name, fallback):
        if len(rows) < 3:
            result = dict(DEFAULTS)
            result["sector_name"] = sector_name
            return result
        erg_vals = sorted(r["erg"] for r in rows)

        def _pctile(arr, pct):
            idx = (pct / 100) * (len(arr) - 1)
            lo  = int(idx)
            hi  = min(lo + 1, len(arr) - 1)
            return round(arr[lo] + (arr[hi] - arr[lo]) * (idx - lo), 4)

        return {
            "multiples":   rows,
            "median":      _pctile(erg_vals, 50),
            "p25":         _pctile(erg_vals, 25),
            "p75":         _pctile(erg_vals, 75),
            "p10":         _pctile(erg_vals, 10),
            "p90":         _pctile(erg_vals, 90),
            "peer_count":  len(erg_vals),
            "sector_name": sector_name,
            "fallback":    fallback,
        }

    # ── Tier 1: industry peers ─────────────────────────────────────
    if industry:
        try:
            _, df = (
                Query()
                .select(*ERG_FIELDS)
                .where(
                    col("industry") == industry,
                    col("market_cap_basic") > MIN_MKTCAP,
                    col("total_revenue_yoy_growth_ttm") > MIN_GROWTH,
                )
                .order_by("market_cap_basic", ascending=False)
                .limit(300)
                .get_scanner_data()
            )
            rows = _extract_rows(df)
            if len(rows) >= MIN_PEERS:
                return _build_stats(rows, industry, False)
        except Exception:
            pass

    # ── Tier 2: sector peers ───────────────────────────────────────
    if sector:
        try:
            _, df = (
                Query()
                .select(*ERG_FIELDS)
                .where(
                    col("sector") == sector,
                    col("market_cap_basic") > MIN_MKTCAP,
                    col("total_revenue_yoy_growth_ttm") > MIN_GROWTH,
                )
                .order_by("market_cap_basic", ascending=False)
                .limit(300)
                .get_scanner_data()
            )
            rows = _extract_rows(df)
            if len(rows) >= MIN_PEERS:
                return _build_stats(rows, "{} (sector)".format(sector), False)
        except Exception:
            pass

    # ── Tier 3: broad-market fallback ─────────────────────────────
    try:
        _, df = (
            Query()
            .select(*ERG_FIELDS)
            .where(
                col("market_cap_basic") > 2e9,
                col("total_revenue_yoy_growth_ttm") > MIN_GROWTH,
            )
            .order_by("market_cap_basic", ascending=False)
            .limit(400)
            .get_scanner_data()
        )
        rows = _extract_rows(df)
        if industry:
            label = "Broad Market (industry '{}' too thin)".format(industry)
        elif sector:
            label = "Broad Market (sector fallback)"
        else:
            label = "Broad Market"
        return _build_stats(rows, label, True)
    except Exception:
        pass

    return DEFAULTS


def fetch_wacc_inputs(ticker: str) -> dict:
    """
    Fetch WACC inputs from yfinance:
      - interest_expense  : TTM interest expense (positive number)
      - income_tax_expense: TTM income tax paid
      - pretax_income     : TTM pre-tax income
      - total_debt_yf     : most recent balance sheet total debt

    Pulls TTM by summing 4 most recent quarters from quarterly_income_stmt.
    Falls back to most-recent annual if fewer than 4 quarters available.
    Returns a dict; any value may be None if unavailable.
    """
    result = {
        "interest_expense":   None,
        "income_tax_expense": None,
        "pretax_income":      None,
        "total_debt_yf":      None,
        "beta_yf":            None,   # 5-year monthly beta (industry standard for WACC)
    }

    if not _YF_AVAILABLE:
        print("  [WACC] yfinance not installed — using fallback values."
              " Fix: pip install yfinance")
        return result

    try:
        ticker_yf = ticker.split(":")[-1] if ":" in ticker else ticker
        tk = yf.Ticker(ticker_yf)

        # ── helpers ──────────────────────────────────────────────────────────
        def _first(df, candidates):
            """Return first matching row name found in df.index, or None."""
            if df is None or df.empty:
                return None
            for name in candidates:
                if name in df.index:
                    return name
            return None

        def _ttm(df, candidates):
            """Sum 4 most recent quarters for the first matching row."""
            key = _first(df, candidates)
            if key is None:
                return None
            row = df.loc[key].dropna()
            vals = row.iloc[:4]
            if len(vals) == 0:
                return None
            return float(vals.sum())

        def _annual(df, candidates):
            """Return most recent annual value for the first matching row."""
            key = _first(df, candidates)
            if key is None:
                return None
            row = df.loc[key].dropna()
            if len(row) == 0:
                return None
            return float(row.iloc[0])

        # ── income statement ─────────────────────────────────────────────────
        # yfinance row names vary by version; try multiple aliases
        IE_NAMES  = ["Interest Expense", "Interest Expense Non Operating",
                     "Net Interest Income", "InterestExpense"]
        TAX_NAMES = ["Tax Provision", "Income Tax Expense",
                     "IncomeTaxExpense", "Provision For Income Taxes"]
        PRE_NAMES = ["Pretax Income", "Income Before Tax",
                     "EarningsBeforeTax", "PretaxIncome"]

        q_inc = tk.quarterly_income_stmt
        use_quarterly = (q_inc is not None and not q_inc.empty
                         and q_inc.shape[1] >= 4)

        if use_quarterly:
            ie  = _ttm(q_inc, IE_NAMES)
            tax = _ttm(q_inc, TAX_NAMES)
            pre = _ttm(q_inc, PRE_NAMES)
            src = "TTM (4Q sum)"
        else:
            a_inc = tk.income_stmt
            ie  = _annual(a_inc, IE_NAMES)
            tax = _annual(a_inc, TAX_NAMES)
            pre = _annual(a_inc, PRE_NAMES)
            src = "annual (most recent FY)"

        # Interest expense is reported as negative outflow — make positive
        if ie is not None:
            ie = abs(ie)

        # ── balance sheet: total debt ─────────────────────────────────────────
        DEBT_NAMES = ["Total Debt", "Long Term Debt And Capital Lease Obligation",
                      "Long Term Debt", "TotalDebt"]
        bs = tk.balance_sheet
        debt_yf = _annual(bs, DEBT_NAMES)

        result["interest_expense"]   = ie
        result["income_tax_expense"] = tax
        result["pretax_income"]      = pre
        result["total_debt_yf"]      = debt_yf

        # Beta from yfinance uses 5-year monthly regression vs S&P500 —
        # the industry standard for WACC (avoids 1-year noise).
        try:
            info = tk.info
            beta_yf = info.get("beta")
            if beta_yf is not None:
                beta_yf = float(beta_yf)
                if not (0.1 <= beta_yf <= 4.0):   # sanity bounds
                    beta_yf = None
            result["beta_yf"] = beta_yf
        except Exception:
            beta_yf = None

        print("  [WACC yfinance {src}] interest={ie}  tax={tax}  "
              "pretax={pre}  debt={dbt}  beta_5y={beta}".format(
            src=src,
            ie  = "{:,.0f}".format(ie)       if ie       is not None else "N/A",
            tax = "{:,.0f}".format(tax)      if tax      is not None else "N/A",
            pre = "{:,.0f}".format(pre)      if pre      is not None else "N/A",
            dbt = "{:,.0f}".format(debt_yf)  if debt_yf  is not None else "N/A",
            beta= "{:.3f}".format(beta_yf)   if beta_yf  is not None else "N/A",
        ))

    except Exception as e:
        print("  [WACC] yfinance fetch failed: " + str(e))

    return result


def fetch_yfinance_extended(ticker: str) -> dict:
    """
    Fetch extended yfinance data needed by the new valuation models.
    All fields default to None on failure; caller must guard accordingly.
    Called in main() immediately after fetch_tv_data().
    """
    import time as _time
    wall_start = _time.time()

    ext = {
        "book_value_ps":            None,
        "stockholders_equity":      None,
        "total_current_assets":     None,
        "total_current_liabilities":None,
        "total_liabilities":        None,
        "roic":                     None,
        "roe":                      None,
        "invested_capital":         None,
        "dividends_per_share":      None,
        "dividend_growth_rate":     None,
        "hist_pe_5y":               None,
        "hist_pfcf_5y":             None,
        "hist_eveb_5y":             None,
        "earnings_surprise_pct":    None,
    }

    if not _YF_AVAILABLE:
        return ext

    try:
        ticker_yf = ticker.split(":")[-1] if ":" in ticker else ticker
        tk = yf.Ticker(ticker_yf)

        def _first(df, candidates):
            if df is None or df.empty:
                return None
            for name in candidates:
                if name in df.index:
                    return name
            return None

        def _annual_val(df, candidates):
            key = _first(df, candidates)
            if key is None:
                return None
            row = df.loc[key].dropna()
            if len(row) == 0:
                return None
            return float(row.iloc[0])

        # ── Info ──────────────────────────────────────────────────────
        try:
            info = tk.info
            bv = info.get("bookValue")
            if bv is not None:
                ext["book_value_ps"] = float(bv)
            div = info.get("dividendRate")
            if div is not None and float(div) > 0:
                ext["dividends_per_share"] = float(div)
        except Exception:
            pass

        # ── Balance sheet ─────────────────────────────────────────────
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                equity = _annual_val(bs, [
                    "Stockholders Equity", "Common Stock Equity",
                    "CommonStockEquity", "Total Equity Gross Minority Interest",
                ])
                if equity is not None:
                    ext["stockholders_equity"] = equity

                cur_assets = _annual_val(bs, [
                    "Current Assets", "Total Current Assets",
                    "Current Assets Total", "TotalCurrentAssets",
                ])
                if cur_assets is not None:
                    ext["total_current_assets"] = cur_assets

                cur_liab = _annual_val(bs, [
                    "Current Liabilities", "Total Current Liabilities",
                    "Current Liabilities Total", "TotalCurrentLiabilities",
                ])
                if cur_liab is not None:
                    ext["total_current_liabilities"] = cur_liab

                tot_liab = _annual_val(bs, [
                    "Total Liabilities Net Minority Interest",
                    "Total Liabilities",
                    "Liabilities",
                ])
                # Fallback: total_assets - stockholders_equity
                if tot_liab is None:
                    tot_assets = _annual_val(bs, ["Total Assets", "Assets"])
                    eq_val = ext.get("stockholders_equity")
                    if tot_assets is not None and eq_val is not None:
                        tot_liab = tot_assets - eq_val
                if tot_liab is not None:
                    ext["total_liabilities"] = tot_liab
        except Exception:
            pass

        # ── ROIC / ROE ────────────────────────────────────────────────
        try:
            cf = tk.cash_flow
            ni_ann = None
            if cf is not None and not cf.empty:
                pass
            inc = tk.income_stmt
            if inc is not None and not inc.empty:
                ni_ann = _annual_val(inc, ["Net Income", "Net Income Common Stockholders"])
            if ni_ann is not None:
                bs2 = tk.balance_sheet
                eq2 = ext["stockholders_equity"]
                dbt2 = _annual_val(bs2, ["Total Debt", "Long Term Debt And Capital Lease Obligation"]) if bs2 is not None and not bs2.empty else None
                cash2 = _annual_val(bs2, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]) if bs2 is not None and not bs2.empty else None
                if eq2 is not None and eq2 > 0:
                    ext["roe"] = ni_ann / eq2
                if eq2 is not None and dbt2 is not None:
                    ic = (dbt2 or 0) + eq2 - (cash2 or 0)
                    ext["invested_capital"] = ic
                    if ic > 0 and ni_ann is not None:
                        ext["roic"] = ni_ann / ic
        except Exception:
            pass

        # ── Dividend growth rate (5-yr CAGR) ──────────────────────────
        try:
            divs = tk.dividends
            if divs is not None and len(divs) >= 2:
                import pandas as _pd
                divs_annual = divs.resample("Y").sum()
                if len(divs_annual) >= 5:
                    d_new = float(divs_annual.iloc[-1])
                    d_old = float(divs_annual.iloc[-6]) if len(divs_annual) >= 6 else float(divs_annual.iloc[0])
                    if d_old > 0 and d_new > 0:
                        n_yrs = min(5, len(divs_annual) - 1)
                        ext["dividend_growth_rate"] = (d_new / d_old) ** (1.0 / n_yrs) - 1.0
        except Exception:
            pass

        # ── Historical multiples ───────────────────────────────────────
        # Use a fresh timer budgeted per-operation (not the global wall_start)
        # so earlier fetches (info, balance sheet, income_stmt, dividends)
        # don't consume the time budget for history.
        try:
            _hist_t0 = _time.time()
            hist_prices = tk.history(period="5y")
            if hist_prices is not None and len(hist_prices) > 50 and _time.time() - _hist_t0 < 20.0:
                    inc5  = tk.income_stmt
                    cf5   = tk.cash_flow
                    bs5   = tk.balance_sheet
                    shares_out = d.get("shares") if isinstance(d, dict) else ext.get("shares")

                    pe_vals, pfcf_vals, eveb_vals = [], [], []

                    if inc5 is not None and not inc5.empty and shares_out:
                        for col_date in inc5.columns:
                            yr = col_date.year
                            yr_prices = hist_prices[hist_prices.index.year == yr]["Close"]
                            if len(yr_prices) < 20:
                                continue
                            avg_price = float(yr_prices.mean())

                            # P/E
                            for k in ["Diluted EPS", "Basic EPS", "EPS"]:
                                if k in inc5.index:
                                    v = inc5.loc[k, col_date]
                                    if v is not None and float(v) > 0.5:
                                        pe = avg_price / float(v)
                                        if 4 < pe < 200:
                                            pe_vals.append(pe)
                                        break

                            # P/FCF
                            if cf5 is not None and not cf5.empty and col_date in cf5.columns:
                                for k in ["Free Cash Flow"]:
                                    if k in cf5.index:
                                        fcf_v = cf5.loc[k, col_date]
                                        if fcf_v is not None and float(fcf_v) > 0:
                                            fcf_ps = float(fcf_v) / shares_out
                                            if fcf_ps > 0:
                                                pfcf = avg_price / fcf_ps
                                                if 4 < pfcf < 200:
                                                    pfcf_vals.append(pfcf)
                                        break

                            # EV/EBITDA — need EBITDA and approximate EV
                            if inc5 is not None and col_date in inc5.columns:
                                ebitda_v = None
                                for k in ["EBITDA", "Normalized EBITDA"]:
                                    if k in inc5.index:
                                        v = inc5.loc[k, col_date]
                                        if v is not None:
                                            ebitda_v = float(v)
                                        break
                                if ebitda_v and ebitda_v > 0 and bs5 is not None and not bs5.empty and col_date in bs5.columns:
                                    debt_v = 0.0
                                    for k in ["Total Debt", "Long Term Debt And Capital Lease Obligation"]:
                                        if k in bs5.index:
                                            v = bs5.loc[k, col_date]
                                            if v is not None:
                                                debt_v = float(v)
                                            break
                                    cash_v = 0.0
                                    for k in ["Cash Cash Equivalents And Short Term Investments",
                                              "Cash And Cash Equivalents"]:
                                        if k in bs5.index:
                                            v = bs5.loc[k, col_date]
                                            if v is not None:
                                                cash_v = float(v)
                                            break
                                    ev_approx = avg_price * shares_out + debt_v - cash_v
                                    if ev_approx > 0:
                                        eveb = ev_approx / ebitda_v
                                        if 2 < eveb < 100:
                                            eveb_vals.append(eveb)

                    if pe_vals:
                        ext["hist_pe_5y"]    = round(sum(pe_vals)   / len(pe_vals),   1)
                    if pfcf_vals:
                        ext["hist_pfcf_5y"]  = round(sum(pfcf_vals) / len(pfcf_vals), 1)
                    if eveb_vals:
                        ext["hist_eveb_5y"]  = round(sum(eveb_vals) / len(eveb_vals), 1)
        except Exception:
            pass

        # ── Earnings surprise ──────────────────────────────────────────
        try:
            eh = tk.earnings_history
            if eh is not None and not eh.empty and len(eh) >= 1:
                last = eh.iloc[0]
                actual = last.get("epsActual") or last.get("Reported EPS")
                est = last.get("epsEstimate") or last.get("EPS Estimate")
                if actual is not None and est is not None and abs(float(est)) > 0.001:
                    ext["earnings_surprise_pct"] = (float(actual) - float(est)) / abs(float(est)) * 100.0
        except Exception:
            pass

    except Exception as e:
        print("  [ext] fetch_yfinance_extended failed: " + str(e))

    print("  [ext] book={bv}  ROIC={roic}  divs={divs}  eq={eq}".format(
        bv   = "{:.2f}".format(ext["book_value_ps"]) if ext["book_value_ps"] else "N/A",
        roic = "{:.1%}".format(ext["roic"])           if ext["roic"] is not None else "N/A",
        divs = "{:.2f}".format(ext["dividends_per_share"]) if ext["dividends_per_share"] else "N/A",
        eq   = "{:,.0f}".format(ext["stockholders_equity"]) if ext["stockholders_equity"] else "N/A",
    ))
    return ext


# ─────────────────────────────────────────────────────────────────
#  WACC CALCULATOR
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  VALUATION ENGINES
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  v4 GROWTH VALUATION ENGINES
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  RELIABILITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────

def B(n):
    if n is None: return "N/A"
    return "$" + format(abs(n) / 1e9, ",.2f") + "B"

def M(n):
    if n is None: return "N/A"
    return "$" + format(abs(n) / 1e6, ",.1f") + "M"

def pc(n):
    return format(n * 100, ".1f") + "%" if n is not None else "N/A"

def mo(n):
    if n is None: return "N/A"
    return "$" + format(n, ",.2f")

def xf(n):
    return format(n, ".1f") + "x" if n is not None else "N/A"

def ud(fv, price):
    if not price or not fv: return ("N/A", "neutral")
    d = (fv - price) / price * 100
    label = ("+" if d >= 0 else "") + format(d, ".1f") + "%"
    color = "up" if d >= 0 else "down"
    return (label, color)




def _build_growth_interpretation(d, gr, reliability, price):
    """
    Build the multi-method interpretation section for the Growth Analysis tab.
    Collects fair values from all available growth methods, then interprets the
    degree of convergence / divergence and what it means for the stock.
    """
    import statistics as _st

    # ── Collect fair values and metadata ─────────────────────────────
    METHOD_COLORS = {
        "Reverse DCF": "#e87c3e",
        "Fwd PEG":     "#f0d500",
        "EV/NTM Rev":  "#00e5c8",
        "TAM Scenario":"#c45aff",
        "Rule of 40":  "#ff7aa2",
        "ERG":         "#38bdf8",
    }

    fvs = {}
    rdcf = gr.get("reverse_dcf")
    if rdcf:
        for s in rdcf.get("scenarios", []):
            if s["label"] == "Street" and s.get("fv", 0) > 0:
                fvs["Reverse DCF"] = s["fv"]; break
    for key, lbl in [("forward_peg","Fwd PEG"),("ev_ntm_revenue","EV/NTM Rev"),
                      ("tam_scenario","TAM Scenario"),("rule_of_40","Rule of 40"),
                      ("erg","ERG")]:
        r = gr.get(key)
        if r and r.get("fair_value", 0) > 0:
            fvs[lbl] = r["fair_value"]

    if len(fvs) < 2:
        return ""

    vals = list(fvs.values())
    median = _st.median(vals)
    mean   = sum(vals) / len(vals)
    lo     = min(vals)
    hi     = max(vals)
    spread_pct = (hi - lo) / median * 100 if median > 0 else 0

    # Convergence: methods within 20% of median
    conv_names  = [m for m, v in fvs.items() if median > 0 and abs(v - median) / median <= 0.20]
    divg_names  = [m for m, v in fvs.items() if median > 0 and abs(v - median) / median > 0.20]
    n_conv      = len(conv_names)
    n_total     = len(vals)

    # Classify convergence quality
    if n_conv == n_total:
        conv_level, conv_clr, conv_label = "HIGH", "#00c896", "Strong Consensus"
    elif n_conv >= n_total * 0.66:
        conv_level, conv_clr, conv_label = "MODERATE", "#f0a500", "Moderate Consensus"
    else:
        conv_level, conv_clr, conv_label = "LOW", "#e05c5c", "Wide Divergence"

    upside = (mean - price) / price * 100 if price and mean else 0
    up_clr = "#00c896" if upside >= 0 else "#e05c5c"
    up_str = ("+" if upside >= 0 else "") + format(upside, ".1f") + "%"

    # ── Narrative interpretation ──────────────────────────────────────
    # 1. Overall signal
    if upside > 20 and n_conv >= n_total * 0.60:
        signal_text = (
            "The growth methods collectively point to meaningful <strong style='color:#00c896;'>upside</strong>. "
            "With {n_conv} of {n_total} methods converging near {mean}, the evidence base is reasonably broad. "
            "The mean implied upside of <strong style='color:{up_clr};'>{up_str}</strong> suggests the market "
            "may be underpricing the company's future earnings power relative to its growth trajectory."
        ).format(n_conv=n_conv, n_total=n_total, mean="$"+format(mean,",.2f"),
                 up_clr=up_clr, up_str=up_str)
    elif upside < -15 and n_conv >= n_total * 0.60:
        signal_text = (
            "The growth methods collectively suggest the stock is <strong style='color:#e05c5c;'>richly priced</strong> "
            "relative to fundamentals. With {n_conv} of {n_total} methods converging below the current price, "
            "the mean implied return of <strong style='color:{up_clr};'>{up_str}</strong> indicates the current "
            "valuation bakes in a very optimistic growth scenario that leaves little margin of safety."
        ).format(n_conv=n_conv, n_total=n_total, up_clr=up_clr, up_str=up_str)
    elif spread_pct > 80:
        signal_text = (
            "The growth methods show <strong style='color:#e05c5c;'>wide disagreement</strong> — the range from "
            "low to high estimate spans <strong>{sp:.0f}%</strong> of the median value. "
            "This is typical of companies where the outcome depends heavily on uncertain assumptions: "
            "growth durability, margin trajectory, and terminal multiple. "
            "No single fair value should be anchored to with high confidence here."
        ).format(sp=spread_pct)
    else:
        signal_text = (
            "The growth methods give a <strong style='color:{up_clr};'>mixed picture</strong>, with a mean implied "
            "return of <strong style='color:{up_clr};'>{up_str}</strong>. "
            "{n_conv} of {n_total} methods converge near {mean}. "
            "The spread between bear and bull estimates is moderate, suggesting some clarity on the fundamental "
            "inputs while still leaving room for differing views on growth sustainability and margin expansion."
        ).format(up_clr=up_clr, up_str=up_str, n_conv=n_conv, n_total=n_total,
                 mean="$"+format(median,",.2f"))

    # 2. Divergence explanations
    divg_items_html = ""
    for mname in divg_names:
        fv = fvs[mname]
        diff_pct = (fv - median) / median * 100
        direction = "above" if diff_pct > 0 else "below"
        abs_diff = abs(diff_pct)
        clr = METHOD_COLORS.get(mname, "#aaa")

        if mname == "Reverse DCF":
            if fv > median:
                interp = ("Reverse DCF is the most bullish here — the market has priced in growth that exceeds "
                          "analyst estimates, or else the stock is genuinely undervalued on a FCF basis. "
                          "Check whether implied growth ({:.0f}%+ higher than street) is realistic.".format(abs_diff))
            else:
                interp = ("Reverse DCF is the most conservative — the market's implied FCF growth assumption is "
                          "modest relative to what other methods require. This may indicate that FCF is currently "
                          "depressed by heavy reinvestment, making the reverse DCF understate long-run value.")
        elif mname == "Fwd PEG":
            if fv > median:
                interp = ("Fwd PEG is the most optimistic method. PEG-based valuation rewards high EPS growth rates "
                          "aggressively — if the growth estimate is reliable, this may be justified; if it's overstated, "
                          "the PEG fair value can become the least trustworthy estimate.")
            else:
                interp = ("Fwd PEG is more pessimistic than the group. This typically occurs when EPS growth is "
                          "expected to slow, or when EPS is currently elevated (making NTM EPS hard to beat). "
                          "Consider whether current EPS reflects the company's true earnings trajectory.")
        elif mname == "EV/NTM Rev":
            if fv > median:
                interp = ("EV/NTM Revenue is pointing higher — the method is market-anchored and may be reflecting "
                          "high current revenue multiples in the stock's sector. This is a signal that the market "
                          "is pricing in sustained growth premium; not necessarily that the stock is cheap.")
            else:
                interp = ("EV/NTM Revenue is more bearish than other methods. Revenue multiples tend to be conservative "
                          "for companies that are scaling margins rapidly — as profitability improves, earnings-based "
                          "methods (PEG, ERG) will tend to pull ahead of revenue multiples.")
        elif mname == "TAM Scenario":
            if fv > median:
                interp = ("TAM Scenario gives the highest estimate — the long-horizon market-share model is capturing "
                          "a larger total addressable market or more aggressive share assumptions. Useful as a ceiling "
                          "but should be weighted less heavily than near-term methods.")
            else:
                interp = ("TAM Scenario is the most conservative method here. This can occur when current revenue "
                          "already represents a substantial fraction of the estimated TAM, limiting upside from "
                          "market-share expansion. It may also reflect a conservative TAM estimate.")
        elif mname == "Rule of 40":
            if fv > median:
                interp = ("Rule of 40 is the most bullish. The company's combined growth and margin score places it "
                          "in a cohort that commands premium EV/Revenue multiples — suggesting the stock may be "
                          "mispriced relative to its SaaS/software peers on a quality-adjusted basis.")
            else:
                interp = ("Rule of 40 is more conservative than the group. A below-cohort score implies the market "
                          "may be awarding the stock a premium that its growth + margin combination doesn't fully "
                          "justify relative to peers. Monitor whether the score is trending up or down.")
        elif mname == "ERG":
            if fv > median:
                interp = ("ERG (Earnings Revenue Growth) is the most bullish. This method is explicitly designed for "
                          "companies whose current margins understate terminal earning power — if the margin expansion "
                          "assumption holds, ERG captures upside other methods miss. High ERG vs. low revenue-multiple "
                          "methods signals: 'buy the margin expansion, not today's revenue multiple'.")
            else:
                interp = ("ERG is the most conservative, despite projecting margin expansion. This typically happens "
                          "when the revenue-multiple methods (EV/NTM Rev, Rule of 40) reflect a currently high market "
                          "multiple that implies more upside than a fundamental earnings build-up supports. "
                          "It can also indicate that current revenue growth is not expected to translate into "
                          "proportionally higher earnings at maturity.")
        else:
            if fv > median:
                interp = "This method is the most bullish. Review its inputs for potential optimism bias."
            else:
                interp = "This method is the most conservative. Review its inputs for potential pessimism or model mis-fit."

        divg_items_html += (
            "<div style='background:var(--surface2);border-left:3px solid {clr};"
            "border-radius:6px;padding:14px 16px;margin-bottom:10px;'>"
            "<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>"
            "<span style='font-weight:700;color:{clr};font-size:14px;'>{mname}</span>"
            "<span style='font-family:\"DM Mono\",monospace;font-size:12px;color:{upclr};'>"
            "${fv} &nbsp;·&nbsp; {sign}{diff:.0f}% {direction} group median</span>"
            "</div>"
            "<p style='font-size:13px;color:var(--muted);margin:0;line-height:1.6;'>{interp}</p>"
            "</div>"
        ).format(
            clr=clr, mname=mname,
            fv=format(fv, ",.2f"),
            upclr="#00c896" if diff_pct > 0 else "#e05c5c",
            sign="+" if diff_pct > 0 else "",
            diff=abs_diff, direction=direction, interp=interp,
        )

    if not divg_items_html:
        divg_items_html = (
            "<p style='color:var(--muted);font-size:13px;'>All methods converge within 20% of the group median — "
            "no significant divergence to explain.</p>"
        )

    # 3. Key risks / tailwinds interpretation
    # Look for patterns across the full set
    notes = []
    rdcf_r = gr.get("reverse_dcf")
    fpeg_r = gr.get("forward_peg")
    erg_r  = gr.get("erg")
    ro40_r = gr.get("rule_of_40")

    if rdcf_r and fpeg_r:
        ig = rdcf_r.get("implied_growth", 0)
        sg = rdcf_r.get("street_growth", 0)
        if ig > sg * 1.20:
            notes.append(
                "📈 <strong>Market pricing in above-consensus growth</strong> — the price implies {:.0f}% annual FCF growth, "
                "vs analyst estimates of {:.0f}%. The stock could underperform if growth disappoints even modestly."
                .format(ig * 100, sg * 100))
        elif ig < sg * 0.80:
            notes.append(
                "📉 <strong>Market pricing in below-consensus growth</strong> — the price implies only {:.0f}% FCF growth "
                "vs analyst estimates of {:.0f}%. If management delivers on guidance, multiple expansion could follow."
                .format(ig * 100, sg * 100))

    if erg_r and fpeg_r:
        erg_fv = erg_r.get("fair_value", 0)
        peg_fv = fpeg_r.get("fair_value", 0)
        if erg_fv > 0 and peg_fv > 0:
            ratio = erg_fv / peg_fv
            if ratio > 1.40:
                notes.append(
                    "💡 <strong>ERG bullish vs PEG</strong> — ERG ({}) exceeds Fwd PEG ({}) by {:.0f}%. "
                    "This typically signals a company where margin expansion is the primary investment thesis: "
                    "top-line growth is repricing the margin structure, so earnings-based PEG understates the story."
                    .format("$"+format(erg_fv,",.2f"), "$"+format(peg_fv,",.2f"), (ratio-1)*100))
            elif ratio < 0.70:
                notes.append(
                    "⚠ <strong>PEG bullish vs ERG</strong> — Fwd PEG ({}) exceeds ERG ({}) by {:.0f}%. "
                    "This can occur when near-term EPS is expected to grow faster than long-run revenue-driven earnings — "
                    "check if the EPS jump is structural or driven by one-time items / buybacks."
                    .format("$"+format(peg_fv,",.2f"), "$"+format(erg_fv,",.2f"), (1/ratio-1)*100))

    if ro40_r:
        ro40_val = ro40_r.get("ro40", 0)
        if ro40_val >= 60:
            notes.append(
                "⭐ <strong>Elite Rule-of-40 score ({:.0f})</strong> — the company is among the top-tier "
                "in combined growth + profitability. Historically, companies sustaining a score ≥60 have "
                "commanded durable premium multiples. Watch for signs of score deterioration as the "
                "company matures.".format(ro40_val))
        elif ro40_val < 20:
            notes.append(
                "⚠ <strong>Below-average Rule-of-40 score ({:.0f})</strong> — growth and profitability together "
                "are below the threshold most institutional investors use for premium SaaS/tech multiples. "
                "Valuation could compress further if the score does not improve.".format(ro40_val))

    # Check flagged methods
    flagged_methods = [m for m in fvs if reliability.get(m, {}).get("flags")]
    if flagged_methods:
        notes.append(
            "🔍 <strong>Low-confidence methods: {}</strong> — reliability flags were raised for these methods "
            "(see cards above). Weight their fair values accordingly.".format(", ".join(flagged_methods)))

    notes_html = ""
    for note in notes:
        notes_html += (
            "<div style='border-left:3px solid var(--border);padding:10px 14px;"
            "margin-bottom:8px;font-size:13px;color:var(--muted);line-height:1.6;'>"
            + note + "</div>"
        )
    if not notes_html:
        notes_html = "<p style='color:var(--muted);font-size:13px;'>No notable cross-method patterns detected.</p>"

    # ── Assemble HTML ─────────────────────────────────────────────────
    return (
        "<div style='margin-top:36px;'>"
        "<div class='section-label'>Growth Methods — Convergence &amp; Divergence Interpretation</div>"
        "<div class='g-note'>"
        "Six growth-oriented valuation methods have been run. This section interprets what their collective "
        "agreement or disagreement implies about the stock's risk/reward profile — not just the average number, "
        "but <em>why</em> the methods agree or diverge and what that means for your investment thesis."
        "</div>"

        # Summary row
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
        "gap:14px;margin-bottom:24px;'>"
        "<div class='range-item'><span class='range-label'>Methods run</span>"
        "<span class='range-value' style='color:var(--text);font-size:22px;'>{ntot}</span></div>"
        "<div class='range-item'><span class='range-label'>Converging (±20%)</span>"
        "<span class='range-value' style='color:{cc};font-size:22px;'>{nconv}/{ntot}</span></div>"
        "<div class='range-item'><span class='range-label'>Mean fair value</span>"
        "<span class='range-value' style='color:var(--text);font-size:22px;'>${mean}</span></div>"
        "<div class='range-item'><span class='range-label'>Mean implied return</span>"
        "<span class='range-value' style='color:{up_clr};font-size:22px;'>{up_str}</span></div>"
        "<div class='range-item'><span class='range-label'>Est. range</span>"
        "<span class='range-value' style='color:var(--text);font-size:16px;'>${lo} – ${hi}</span></div>"
        "</div>"

        # Convergence badge
        "<div style='background:rgba({rgb},0.10);border:1px solid rgba({rgb},0.30);"
        "border-radius:10px;padding:20px 24px;margin-bottom:24px;'>"
        "<div style='font-size:13px;font-weight:700;color:{cc};letter-spacing:1px;margin-bottom:10px;'>"
        "{conv_level} CONVICTION &nbsp;·&nbsp; {conv_label}</div>"
        "<p style='font-size:13px;color:var(--muted);margin:0;line-height:1.7;'>{signal_text}</p>"
        "</div>"

        # Divergence details
        "<div style='margin-bottom:24px;'>"
        "<div style='font-size:11px;font-weight:700;color:var(--muted);letter-spacing:2px;"
        "text-transform:uppercase;margin-bottom:12px;'>Outlier Methods &amp; Why They Differ</div>"
        "{divg_items}"
        "</div>"

        # Cross-method patterns
        "<div style='margin-bottom:24px;'>"
        "<div style='font-size:11px;font-weight:700;color:var(--muted);letter-spacing:2px;"
        "text-transform:uppercase;margin-bottom:12px;'>Cross-Method Patterns &amp; Signals</div>"
        "{notes}"
        "</div>"
        "</div>"
    ).format(
        ntot=n_total, nconv=n_conv, mean=format(mean,",.2f"),
        up_clr=up_clr, up_str=up_str, cc=conv_clr,
        lo=format(lo,",.2f"), hi=format(hi,",.2f"),
        rgb="0,200,150" if conv_level=="HIGH" else "240,165,0" if conv_level=="MODERATE" else "224,92,92",
        conv_level=conv_level, conv_label=conv_label,
        signal_text=signal_text, divg_items=divg_items_html, notes=notes_html,
    )


def _build_growth_html(d, gr, reliability=None):
    """Render the Growth Analysis tab HTML from computed growth method results."""
    reliability = reliability or {}
    price=d["price"]
    ticker=d["ticker"]
    company_name=d.get("company_name", ticker)
    B2=lambda n:("$"+format(abs(n)/1e9,",.2f")+"B") if n is not None else "N/A"
    mo2=lambda n:("$"+format(n,",.2f")) if n is not None else "N/A"

    # ── Build growth summary header ────────────────────────────────
    # Collect fair values from all growth methods that ran successfully
    _gfvs = {}
    _rdcf = gr.get("reverse_dcf")
    if _rdcf:
        for _s in _rdcf.get("scenarios", []):
            if _s["label"] == "Street" and _s.get("fv", 0) > 0:
                _gfvs["Reverse DCF"] = _s["fv"]; break
    for _k, _lbl in [("forward_peg","Fwd PEG"),("ev_ntm_revenue","EV/NTM Rev"),
                      ("tam_scenario","TAM Scenario"),("rule_of_40","Rule of 40"),
                      ("erg","ERG")]:
        _r = gr.get(_k)
        if _r and _r.get("fair_value") and _r["fair_value"] > 0:
            _gfvs[_lbl] = _r["fair_value"]

    _g_method_colors = {
        "Reverse DCF":"#e87c3e","Fwd PEG":"#f0d500",
        "EV/NTM Rev":"#00e5c8","TAM Scenario":"#c45aff","Rule of 40":"#ff7aa2",
        "ERG":"#38bdf8","S-Curve TAM":"#c45aff","PIE":"#ff6b9d","DDM":"#27ae60",
    }

    # Conviction: how many methods agree within 20% of the median growth fair value
    _g_summary_html = ""
    if _gfvs:
        import statistics as _st
        _vals = list(_gfvs.values())
        _med  = _st.median(_vals) if _vals else 0
        _mean = sum(_vals)/len(_vals) if _vals else 0
        _converging_names = [mn for mn,v in _gfvs.items() if _med > 0 and abs(v - _med)/_med <= 0.20]
        _converging = len(_converging_names)
        _total_methods = len(_vals)
        if _converging >= _total_methods:          _gconv, _gcclr = "HIGH",     "#00c896"
        elif _converging >= _total_methods * 0.6:  _gconv, _gcclr = "MODERATE", "#f0a500"
        else:                                       _gconv, _gcclr = "LOW",      "#e05c5c"

        _upside = (_mean - price) / price * 100 if (price and _mean) else 0
        _up_clr = "#00c896" if _upside >= 0 else "#e05c5c"
        _up_str = ("+" if _upside >= 0 else "") + format(_upside, ".1f") + "%"

        # Method pills
        _pills = ""
        _MONO = "DM Mono"
        for _mn, _fv in _gfvs.items():
            _clr = _g_method_colors.get(_mn, "#aaa")
            _u   = (_fv - price) / price * 100 if price else 0
            _us  = ("+" if _u >= 0 else "") + format(_u, ".1f") + "%"
            _uc  = "#00c896" if _u >= 0 else "#e05c5c"
            _pills += (
                "<div style='background:var(--surface2);border:1px solid var(--border);"
                "border-left:3px solid %(c)s;border-radius:8px;padding:10px 14px;min-width:110px;'>"
                "<div style='font-size:10px;color:var(--muted);letter-spacing:1px;margin-bottom:4px;'>%(mn)s</div>"
                "<div style='font-size:18px;font-weight:700;font-family:%(mono)s,monospace;'>$%(fv)s</div>"
                "<div style='font-size:11px;color:%(uc)s;font-family:%(mono)s,monospace;'>%(us)s</div>"
                "</div>"
            ) % {"c":_clr,"mn":_mn,"fv":format(_fv,",.2f"),"uc":_uc,"us":_us,"mono":_MONO}

        _g_summary_html = (
            "<div style='background:var(--surface);border:1px solid var(--border);"
            "border-radius:14px;padding:24px 28px;margin-bottom:28px;'>"
            "<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:6px;'>"
            "<span style='font-size:28px;font-weight:800;color:var(--text);'>%(cn)s</span>"
            "<span style='font-family:DM Mono,monospace;font-size:14px;color:var(--muted);letter-spacing:2px;'>%(tk)s</span>"
            "<span style='font-size:13px;color:var(--muted);margin-left:4px;'>&nbsp;&middot;&nbsp;Current Price: "
            "<strong style='color:var(--text);'>$%(pr)s</strong></span>"
            "</div>"
            "<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:18px;'>"
            "<div style='background:rgba(%(rgb)s,0.15);"
            "border:1px solid rgba(%(rgb)s,0.4);border-radius:8px;padding:10px 16px;'>"
            "<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            "<span style='font-size:12px;font-weight:700;color:%(gcclr)s;letter-spacing:1px;'>%(gconv)s CONVICTION</span>"
            "&nbsp;<span style='font-size:11px;color:var(--muted);'>%(cv)s/%(tot)s methods converge</span>"
            "</div>"
            "<div style='font-size:11px;color:var(--muted);'>Converging: "
            "<span style='color:%(gcclr)s;font-weight:600;'>%(conv_names)s</span></div>"
            "</div>"
            "<div style='font-size:13px;color:var(--muted);'>Growth mean fair value: "
            "<strong style='color:var(--text);'>$%(mean)s</strong> &nbsp;&middot;&nbsp; "
            "Mean upside: <strong style='color:%(upclr)s;'>%(ups)s</strong></div>"
            "</div>"
            "<div style='display:flex;gap:10px;flex-wrap:wrap;'>%(pills)s</div>"
            "</div>"
        ) % {
            "cn":company_name,"tk":ticker,"pr":format(price,",.2f"),
            "gcclr":_gcclr,"gconv":_gconv,"cv":_converging,"tot":_total_methods,
            "conv_names":", ".join(_converging_names) if _converging_names else "none",
            "rgb":"0,200,150" if _gconv=="HIGH" else "240,165,0" if _gconv=="MODERATE" else "224,92,92",
            "mean":format(_mean,",.2f"),"upclr":_up_clr,"ups":_up_str,"pills":_pills,
        }

    def ud(fv):
        if not fv or not price: return("N/A","neutral")
        p=(fv-price)/price*100
        return(("+" if p>=0 else "")+format(p,".1f")+"%","up" if p>=0 else "down")
    def sud(u):
        return(("+" if u>=0 else "")+format(u,".1f")+"%","sup" if u>=0 else "sdn")

    def gflag(method_name):
        """Return flag HTML for a growth method, or empty string if reliable."""
        rel = reliability.get(method_name, {})
        fl = rel.get("flags", [])
        if not fl: return "", ""
        items = "".join('<div class="flag-item">⚠ ' + f + '</div>' for f in fl)
        return '<div class="reliability-flag">' + items + '</div>', ' method-flagged'

    fwd_eps_s=("analyst NTM EPS: <strong>"+mo2(d.get("fwd_eps"))+"</strong>") if d.get("fwd_eps") else "no analyst EPS forecast"
    fwd_rev_s=("analyst NTM Rev: <strong>"+B2(d.get("fwd_rev"))+"</strong>") if d.get("fwd_rev") else "no analyst revenue forecast"
    html=(
        _g_summary_html +
        '<div class="section-label">Growth Analysis — Six Forward-Looking Methods</div>'
        '<div class="g-note">'
        '<strong>Inputs:</strong>&nbsp; Est. growth: <strong>'+format(d["est_growth"]*100,".1f")+'%</strong>'
        ' ('+d.get("growth_source","FCF-margin proxy")+')&nbsp;·&nbsp;'
        +fwd_eps_s+'&nbsp;·&nbsp;'+fwd_rev_s+'&nbsp;·&nbsp;'
        'Gross margin: <strong>'+(format(d["gross_margin"],".1f")+"%" if d.get("gross_margin") else "N/A")+'</strong>'
        '</div>'
        '<div class="g-grid">'
    )

    # 1. Reverse DCF
    rdcf=gr.get("reverse_dcf")
    if rdcf:
        ig=rdcf["implied_growth"]; sg=rdcf["street_growth"]
        diff=(ig-sg)*100; dc="sup" if diff>=0 else "sdn"
        ds=("+" if diff>=0 else "")+format(diff,".1f")+"ppt vs Street"
        rows=""
        for s in rdcf["scenarios"]:
            cls="hi" if s["label"]=="Implied" else ""
            us,uc=sud(s["upside"])
            rows+="<tr class='{c}'><td>{l}</td><td>{g:.1f}%</td><td>{fv}</td><td class='{uc}'>{us}</td></tr>".format(
                c=cls,l=s["label"],g=s["g"]*100,fv=mo2(s["fv"]),uc=uc,us=us)
        fh,fc=gflag("Reverse DCF")
        html+=(
            '<div class="gc'+fc+'" style="--gca:#e87c3e;">'
            '<div class="gc-head"><span class="gc-name">Reverse DCF</span><span class="gc-badge">Growth Implied</span></div>'
            +fh+
            '<div class="gc-sub">What FCF growth rate is baked into today&#39;s price? '
            'Compare to your own conviction. Implied &lt; Street → market is cautious; Implied &gt; Street → priced for perfection.</div>'
            '<div class="ig-box"><div class="ig-num">'+format(ig*100,".1f")+'%</div>'
            '<div class="ig-lbl">implied annual FCF<br>growth over 5 yrs<br>(WACC '+format(rdcf["wacc"]*100,".1f")+'%)</div>'
            '<div class="ig-vs"><div class="ig-vs-k">Street estimate</div><div class="ig-vs-v">'+format(sg*100,".1f")+'%</div>'
            '<div class="ig-vs-k" style="margin-top:5px;">Difference</div>'
            '<div class="ig-vs-v '+dc+'">'+ds+'</div></div></div>'
            '<table class="sc-tbl"><thead><tr><th>Scenario</th><th>Growth</th><th>Fair Value</th><th>Upside</th></tr></thead>'
            '<tbody>'+rows+'</tbody></table></div>'
        )
    else:
        html+='<div class="gc" style="--gca:#e87c3e;"><div class="gc-head"><span class="gc-name">Reverse DCF</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires positive FCF data.</p></div>'

    # 2. Forward PEG
    fpeg=gr.get("forward_peg")
    if fpeg:
        fv=fpeg["fair_value"]; ud_s,ud_c=ud(fv)
        cpeg=(format(fpeg["current_fwd_peg"],".2f")+"x") if fpeg.get("current_fwd_peg") else "N/A"
        fh,fc=gflag("Fwd PEG")
        html+=(
            '<div class="gc'+fc+'" style="--gca:#f0a500;">'
            '<div class="gc-head"><span class="gc-name">Forward PEG</span><span class="gc-badge">Earnings Growth</span></div>'
            +fh+
            '<div class="gc-sub">NTM EPS × (growth% × PEG). Lynch rule: P/E = EPS growth rate is fair value (PEG 1.0). '
            'Quality moats can justify 1.5–2.0×.</div>'
            '<div class="gc-fv">'+mo2(fv)+'</div>'
            '<div class="gc-ud '+ud_c+'">'+ud_s+' vs current price</div>'
            '<div class="gc-mos">MoS Price: '+mo2(fpeg["mos_value"])+'</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>NTM EPS ('+fpeg["eps_source"]+')</td><td>'+mo2(fpeg["ntm_eps"])+'</td></tr>'
            '<tr><td>Est. growth rate</td><td>'+format(fpeg["growth"]*100,".1f")+'%</td></tr>'
            '<tr><td>Current Fwd PEG</td><td>'+cpeg+'</td></tr>'
            '<tr><td>Conservative — PEG 1.0×</td><td>'+mo2(fpeg["fv_conserv"])+'</td></tr>'
            '<tr><td>Base — PEG 1.5×</td><td>'+mo2(fpeg["fv_base"])+'</td></tr>'
            '<tr><td>Premium — PEG 2.0×</td><td>'+mo2(fpeg["fv_premium"])+'</td></tr>'
            '</table></div>'
        )
    else:
        html+='<div class="gc" style="--gca:#f0a500;"><div class="gc-head"><span class="gc-name">Forward PEG</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires positive EPS or analyst forecast.</p></div>'

    # 3. EV / NTM Revenue
    evr=gr.get("ev_ntm_revenue")
    if evr:
        fv=evr["fair_value"]; ud_s,ud_c=ud(fv)
        cev=(format(evr["current_ev_rev"],".1f")+"x") if evr.get("current_ev_rev") else "N/A"
        gms=(format(evr["gross_margin"],".1f")+"%") if evr.get("gross_margin") else "N/A"
        fh,fc=gflag("EV/NTM Rev")
        html+=(
            '<div class="gc'+fc+'" style="--gca:#00c896;">'
            '<div class="gc-head"><span class="gc-name">EV / NTM Revenue</span><span class="gc-badge">Revenue Multiple</span></div>'
            +fh+
            '<div class="gc-sub">Forward revenue × margin-tiered &amp; market-anchored EV/Rev. '
            'Blends fundamental tier ranges with the stock&#39;s current market multiple for grounded estimates.</div>'
            '<div class="gc-fv">'+mo2(fv)+'</div>'
            '<div class="gc-ud '+ud_c+'">'+ud_s+' vs current price</div>'
            '<div class="gc-mos">MoS Price: '+mo2(evr["mos_value"])+'</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>NTM Revenue ('+evr["rev_source"]+')</td><td>'+B2(evr["ntm_rev"])+'</td></tr>'
            '<tr><td>Gross margin</td><td>'+gms+'</td></tr>'
            '<tr><td>Tier</td><td style="font-size:10px;text-align:right;">'+evr["tier"]+'</td></tr>'
            '<tr><td>Growth bonus (to mid)</td><td>+'+format(evr["growth_bonus"],".2f")+'×</td></tr>'
            '<tr><td>Multiple source</td><td style="font-size:10px;text-align:right;">'+evr.get("anchor_source","tier-only")+'</td></tr>'
            '<tr><td>Current EV/Revenue</td><td>'+cev+'</td></tr>'
            '<tr><td>Bear ('+format(evr["mult_lo"],".1f")+'×)</td><td>'+mo2(evr["fv_lo"])+'</td></tr>'
            '<tr><td>Base ('+format(evr["mult_mid"],".1f")+'×)</td><td>'+mo2(evr["fv_mid"])+'</td></tr>'
            '<tr><td>Bull ('+format(evr["mult_hi"],".1f")+'×)</td><td>'+mo2(evr["fv_hi"])+'</td></tr>'
            '</table></div>'
        )
    else:
        html+='<div class="gc" style="--gca:#00c896;"><div class="gc-head"><span class="gc-name">EV / NTM Revenue</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires revenue data.</p></div>'

    # 4. TAM Scenario
    tam=gr.get("tam_scenario")
    if tam:
        fv=tam["fair_value"]; ud_s,ud_c=ud(fv)
        rows=""
        for s in tam["scenarios"]:
            us,uc=sud(s["upside"])
            rows+="<tr><td>{l}</td><td>{sp:.0f}%</td><td>{rv}</td><td>{nm:.1f}%</td><td>{fv}</td><td class='{uc}'>{us}</td></tr>".format(
                l=s["label"],sp=s["share_pct"],rv=B2(s["yr5_rev"]),nm=s["net_margin"],fv=mo2(s["fv"]),uc=uc,us=us)
        fh,fc=gflag("TAM Scenario")
        html+=(
            '<div class="gc'+fc+'" style="--gca:#bf6ff0;">'
            '<div class="gc-head"><span class="gc-name">TAM Scenario</span><span class="gc-badge">Market Share Model</span></div>'
            +fh+
            '<div class="gc-sub">Year-5 revenue from market-share capture → terminal earnings at 25× P/E, discounted back. '
            'Bear/base/bull market-share scenarios are anchored to the company\'s implied year-5 trajectory (from current growth rate), '
            'so the base case reflects where the company is actually headed, not a fixed % cap. '
            'Net margin uses actual TTM data where available, falling back to a gross-margin proxy.</div>'
            '<div class="gc-fv">'+mo2(fv)+'</div>'
            '<div class="gc-ud '+ud_c+'">'+ud_s+' vs current price (base)</div>'
            '<div class="gc-mos">MoS Price: '+mo2(tam["mos_value"])+'</div><hr>'
            '<table class="gc-tbl" style="margin-bottom:10px;">'
            '<tr><td>Est. TAM ('+format(tam["tam_mult"],".0f")+'× current rev)</td><td>'+B2(tam["tam_est"])+'</td></tr>'
            '<tr><td>Base net margin ('+tam.get("nm_source","GM proxy")+')</td><td>'+format(tam["base_net_margin"],".1f")+'%</td></tr>'
            '<tr><td>Implied Yr-5 market share (anchor)</td><td>'+format(tam.get("implied_share_pct",8.0),".1f")+'%</td></tr>'
            '<tr><td>Terminal P/E / WACC</td><td>25× / '+format(tam["wacc"]*100,".1f")+'%</td></tr>'
            '</table>'
            '<table class="sc-tbl"><thead><tr>'
            '<th>Scenario</th><th>Mkt Share</th><th>Yr-5 Rev</th><th>Net Margin</th><th>Fair Value</th><th>Upside</th>'
            '</tr></thead><tbody>'+rows+'</tbody></table></div>'
        )
    else:
        html+='<div class="gc" style="--gca:#bf6ff0;"><div class="gc-head"><span class="gc-name">TAM Scenario</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires revenue data.</p></div>'

    # 5. Rule of 40
    ro=gr.get("rule_of_40")
    if ro:
        fv=ro["fair_value"]; ud_s,ud_c=ud(fv)
        sc=("#00c896" if ro["ro40"]>=60 else "#f0a500" if ro["ro40"]>=40 else "#e05c5c")
        cev=(format(ro["current_ev_rev"],".1f")+"x") if ro.get("current_ev_rev") else "N/A"
        fh,fc=gflag("Rule of 40")
        html+=(
            '<div class="gc'+fc+'" style="--gca:'+sc+';">'
            '<div class="gc-head"><span class="gc-name">Rule of 40</span><span class="gc-badge">Quality + Revenue</span></div>'
            +fh+
            '<div class="gc-sub">Rev growth% + FCF margin% — the SaaS quality benchmark. '
            'Score determines which peer cohort EV/Revenue range applies.</div>'
            '<div class="ro40-num" style="color:'+sc+';">'+format(ro["ro40"],".1f")+'</div>'
            '<div class="ro40-cohort">'+ro["cohort"]
            +'&nbsp;·&nbsp;Rev growth '+format(ro["rev_growth_pct"],".1f")
            +'% + FCF margin '+format(ro["fcf_margin_pct"],".1f")+'%</div>'
            '<div class="gc-fv">'+mo2(fv)+'</div>'
            '<div class="gc-ud '+ud_c+'">'+ud_s+' vs current price (mid cohort)</div>'
            '<div class="gc-mos">MoS Price: '+mo2(ro["mos_value"])+'</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>Current EV / Revenue</td><td>'+cev+'</td></tr>'
            '<tr><td>Cohort EV/Rev range</td><td>'+format(ro["mult_lo"],".0f")+'× – '+format(ro["mult_hi"],".0f")+'×</td></tr>'
            '<tr><td>Bear ('+format(ro["mult_lo"],".0f")+'×)</td><td>'+mo2(ro["fv_lo"])+'</td></tr>'
            '<tr><td>Mid ('+format(ro["mult_mid"],".0f")+'×)</td><td>'+mo2(ro["fv_mid"])+'</td></tr>'
            '<tr><td>Bull ('+format(ro["mult_hi"],".0f")+'×)</td><td>'+mo2(ro["fv_hi"])+'</td></tr>'
            '</table></div>'
        )
    else:
        html+='<div class="gc" style="--gca:#6b7194;"><div class="gc-head"><span class="gc-name">Rule of 40</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires revenue data.</p></div>'

    # 6. ERG — Earnings-based Revenue Growth (dual approach)
    erg=gr.get("erg")
    if erg:
        fv=erg["fair_value"]; ud_s,ud_c=ud(fv)
        fh,fc=gflag("ERG")
        # Approach A scenario table
        rows_a=""
        for s in erg.get("scenarios_a",[]):
            us,uc=sud(s["upside"])
            rows_a+="<tr><td>{l}</td><td>{g:.1f}%</td><td>{nm:.1f}%</td><td>{pe:.0f}×</td><td>{fv}</td><td class='{uc}'>{us}</td></tr>".format(
                l=s["label"],g=s["g_rate"]*100,nm=s["nm"]*100,pe=s["term_pe"],fv=mo2(s["fv"]),uc=uc,us=us)
        # Calibration detail block
        calib = erg.get("erg_calib_details")
        calib_html = ""
        if calib and erg.get("calibrated_erg") is not None:
            calib_html = (
                "<div style='background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.2);"
                "border-radius:8px;padding:12px 14px;margin:10px 0;font-size:12px;'>"
                "<div style='font-size:10px;font-weight:700;color:#38bdf8;letter-spacing:1.5px;"
                "text-transform:uppercase;margin-bottom:8px;'>EV/Rev-to-Growth Calibration</div>"
                "<table style='width:100%;border-collapse:collapse;'>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Peer universe</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;'>"
                "{n} peers — {sec}</td></tr>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Peer ERG (p25/med/p75)</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;'>"
                "{p25:.3f} / {med:.3f} / {p75:.3f}</td></tr>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Subject GM {gm:.1f}%</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;'>"
                "→ {gmp:.0f}th pctile</td></tr>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Subject Ro40 {ro40:.1f}</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;'>"
                "→ {r40p:.0f}th pctile</td></tr>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Composite quality rank</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;color:#38bdf8;font-weight:700;'>"
                "{cp:.0f}th pctile</td></tr>"
                "<tr><td style='color:var(--muted);padding:3px 0;'>Calibrated ERG multiple</td>"
                "<td style='text-align:right;font-family:DM Mono,monospace;color:#38bdf8;font-weight:700;'>"
                "{erg:.3f}×</td></tr>"
                "</table></div>"
            ).format(
                n=calib.get("peer_count",0),
                sec=calib.get("sector_name","N/A"),
                p25=calib.get("peer_erg_p25",0), med=calib.get("peer_erg_median",0), p75=calib.get("peer_erg_p75",0),
                gm=calib.get("subj_gm",0), gmp=calib.get("gm_pctile",50),
                ro40=calib.get("subj_ro40",0), r40p=calib.get("ro40_pctile",50),
                cp=calib.get("composite_pctile",50),
                erg=erg.get("calibrated_erg",0),
            )
            # Approach B summary row
            fv_b = erg.get("fv_b_base")
            if fv_b and fv_b > 0:
                us_b,uc_b = ud(fv_b)
                calib_html += (
                    "<table class='gc-tbl' style='margin-bottom:4px;'>"
                    "<tr><td>EV/Rev-to-Growth (bear p25)</td><td>"+mo2(erg.get("fv_b_bear",0))+"</td></tr>"
                    "<tr><td>EV/Rev-to-Growth (base cal.)</td><td style='color:#38bdf8;font-weight:700;'>"+mo2(fv_b)+"</td></tr>"
                    "<tr><td>EV/Rev-to-Growth (bull p75)</td><td>"+mo2(erg.get("fv_b_bull",0))+"</td></tr>"
                    "</table>"
                )
        else:
            calib_html = (
                "<div style='font-size:11px;color:var(--muted);padding:8px 0;'>"
                "EV/Rev-to-Growth approach unavailable — no peer calibration data. "
                "Earnings Build only.</div>"
            )
        html+=(
            '<div class="gc'+fc+'" style="--gca:#38bdf8;">'
            '<div class="gc-head"><span class="gc-name">ERG</span>'
            '<span class="gc-badge">Dual-Approach Growth</span></div>'
            +fh+
            '<div class="gc-sub">'
            '<strong>Blended fair value: 60% Earnings Build + 40% EV/Rev-to-Growth.</strong> '
            'The Earnings Build projects 5-year revenue, applies a maturity net-margin target, and discounts terminal earnings back at WACC. '
            'The EV/Rev-to-Growth approach derives a live ERG multiple (EV/Revenue ÷ growth%) from '
            'actual sector peers on TradingView, then quality-adjusts it to reflect where this company sits '
            'in the peer distribution on gross margin and Rule-of-40 — eliminating the arbitrary hardcoded multiplier.'
            '</div>'
            '<div class="gc-fv">'+mo2(fv)+'</div>'
            '<div class="gc-ud '+ud_c+'">'+ud_s+' vs current price (blended base)</div>'
            '<div style="font-size:11px;color:var(--muted);margin-top:2px;margin-bottom:6px;">'+erg.get("blend_note","")+'</div>'
            '<div class="gc-mos">MoS Price: '+mo2(erg["mos_value"])+'</div><hr>'
            '<div style="font-size:10px;font-weight:700;color:var(--muted);letter-spacing:1.5px;'
            'text-transform:uppercase;margin:8px 0 6px;">Blended Bear / Base / Bull</div>'
            '<table class="gc-tbl" style="margin-bottom:10px;">'
            '<tr><td>Bear (blended)</td><td>'+mo2(erg["fv_bear"])+'</td></tr>'
            '<tr><td>Base (blended)</td><td style="font-weight:700;">'+mo2(erg["fv_base"])+'</td></tr>'
            '<tr><td>Bull (blended)</td><td>'+mo2(erg["fv_bull"])+'</td></tr>'
            '</table>'
            '<div style="font-size:10px;font-weight:700;color:var(--muted);letter-spacing:1.5px;'
            'text-transform:uppercase;margin:8px 0 6px;">Approach A — Earnings Build</div>'
            '<table class="gc-tbl" style="margin-bottom:6px;">'
            '<tr><td>Growth rate</td><td>'+format(erg["growth"]*100,".1f")+'%</td></tr>'
            '<tr><td>Margin source</td><td style="font-size:10px;text-align:right;">'+erg["nm_source"]+'</td></tr>'
            '<tr><td>Growth tier</td><td style="font-size:10px;text-align:right;">'+erg["tier_label"]+'</td></tr>'
            '<tr><td>WACC</td><td>'+format(erg["wacc"]*100,".1f")+'%</td></tr>'
            '<tr><td>A-Bear: NM '+format(erg["nm_bear"]*100,".1f")+'% at '+format(erg["te_bear"],".0f")+'× P/E</td><td>'+mo2(erg["fv_a_bear"])+'</td></tr>'
            '<tr><td>A-Base: NM '+format(erg["nm_base"]*100,".1f")+'% at '+format(erg["te_base"],".0f")+'× P/E</td><td>'+mo2(erg["fv_a_base"])+'</td></tr>'
            '<tr><td>A-Bull: NM '+format(erg["nm_bull"]*100,".1f")+'% at '+format(erg["te_bull"],".0f")+'× P/E</td><td>'+mo2(erg["fv_a_bull"])+'</td></tr>'
            '</table>'
            '<table class="sc-tbl" style="margin-bottom:10px;"><thead><tr>'
            '<th>Scenario</th><th>Growth</th><th>NM yr-5</th><th>Term P/E</th><th>Fair Value</th><th>Upside</th>'
            '</tr></thead><tbody>'+rows_a+'</tbody></table>'
            '<div style="font-size:10px;font-weight:700;color:var(--muted);letter-spacing:1.5px;'
            'text-transform:uppercase;margin:8px 0 6px;">Approach B — EV/Rev-to-Growth (Peer-Calibrated)</div>'
            +calib_html+
            '</div>'
        )
    else:
        html+='<div class="gc" style="--gca:#38bdf8;"><div class="gc-head"><span class="gc-name">ERG</span></div><p style="color:var(--muted);font-size:13px;margin-top:12px;">Requires revenue data.</p></div>'

    # 7. S-Curve TAM
    sct = gr.get("scurve_tam")
    if sct:
        fv = sct["fair_value"]; ud_s, ud_c = ud(fv)
        fh, fc = gflag("S-Curve TAM")
        scens = sct.get("scenarios", [])
        s_rows = ""
        for sc in scens:
            us, uc = sud((sc["fv"] - price) / price * 100 if price else 0)
            s_rows += "<tr><td>{l}</td><td>{fv}</td><td class='{uc}'>{us}</td></tr>".format(
                l=sc.get("label",""), fv=mo2(sc.get("fv")), uc=uc, us=us)
        html += (
            '<div class="gc' + fc + '" style="--gca:#c45aff;">'
            '<div class="gc-head"><span class="gc-name">S-Curve TAM</span>'
            '<span class="gc-badge">Logistic Growth</span></div>' + fh +
            '<div class="gc-sub">Models adoption via logistic (S-shaped) curve within the total addressable market. '
            'Three scenarios vary the steepness parameter k.</div>'
            '<div class="gc-fv">' + mo2(fv) + '</div>'
            '<div class="gc-ud ' + ud_c + '">' + ud_s + ' vs current price</div>'
            '<div class="gc-mos">MoS Price: ' + mo2(sct.get("mos_value")) + '</div><hr>'
            '<table class="sc-tbl"><thead><tr><th>Scenario</th><th>Fair Value</th><th>Upside</th></tr></thead>'
            '<tbody>' + s_rows + '</tbody></table></div>'
        )
    else:
        html += ('<div class="gc" style="--gca:#c45aff;"><div class="gc-head">'
                 '<span class="gc-name">S-Curve TAM</span></div>'
                 '<p style="color:var(--muted);font-size:13px;margin-top:12px;">'
                 'Requires revenue data and market share below 70%.</p></div>')

    # 8. PIE (Price Implied Expectations)
    pie = gr.get("pie")
    if pie:
        fv = pie["fair_value"]; ud_s, ud_c = ud(fv)
        fh, fc = gflag("PIE")
        verdict = pie.get("verdict", "")
        adj = pie.get("adjustment_factor", 1.0)
        ig = pie.get("implied_growth", 0)
        eg = pie.get("analyst_growth", 0)
        html += (
            '<div class="gc' + fc + '" style="--gca:#ff6b9d;">'
            '<div class="gc-head"><span class="gc-name">Price Implied Expectations</span>'
            '<span class="gc-badge">Market Sentiment</span></div>' + fh +
            '<div class="gc-sub">Reverse-engineers the growth rate baked into today\'s price. '
            'Compares market-implied growth vs analyst consensus to find mis-pricings.</div>'
            '<div class="gc-fv">' + mo2(fv) + '</div>'
            '<div class="gc-ud ' + ud_c + '">' + ud_s + ' vs current price</div>'
            '<div class="gc-mos">MoS Price: ' + mo2(pie.get("mos_value")) + '</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>Implied Revenue Growth</td><td>' + format(ig * 100, ".1f") + '%</td></tr>'
            '<tr><td>Analyst Consensus Growth</td><td>' + format(eg * 100, ".1f") + '%</td></tr>'
            '<tr><td>Adjustment Factor</td><td>' + format(adj, ".2f") + 'x</td></tr>'
            '<tr><td>Verdict</td><td style="color:#ff6b9d;">' + verdict + '</td></tr>'
            '</table></div>'
        )
    else:
        html += ('<div class="gc" style="--gca:#ff6b9d;"><div class="gc-head">'
                 '<span class="gc-name">Price Implied Expectations</span></div>'
                 '<p style="color:var(--muted);font-size:13px;margin-top:12px;">'
                 'Requires price, revenue, and shares data.</p></div>')

    # 9. DDM H-Model
    ddm = gr.get("ddm")
    if ddm:
        fv = ddm["fair_value"]; ud_s, ud_c = ud(fv)
        fh, fc = gflag("DDM")
        html += (
            '<div class="gc' + fc + '" style="--gca:#27ae60;">'
            '<div class="gc-head"><span class="gc-name">DDM H-Model</span>'
            '<span class="gc-badge">Dividend Growth</span></div>' + fh +
            '<div class="gc-sub">Two-stage dividend discount model. Stage 1: high growth phase (H=5 half-life). '
            'Stage 2: terminal growth. Best for dividend-paying stocks.</div>'
            '<div class="gc-fv">' + mo2(fv) + '</div>'
            '<div class="gc-ud ' + ud_c + '">' + ud_s + ' vs current price</div>'
            '<div class="gc-mos">MoS Price: ' + mo2(ddm.get("mos_value")) + '</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>Annual Dividend (D0)</td><td>' + mo2(ddm.get("d0")) + '</td></tr>'
            '<tr><td>Gordon Component</td><td>' + mo2(ddm.get("gordon_component")) + '</td></tr>'
            '<tr><td>H-Model Component</td><td>' + mo2(ddm.get("hmodel_component")) + '</td></tr>'
            '<tr><td>Cost of Equity (Ke)</td><td>' + (format(ddm.get("ke", 0) * 100, ".1f") + "%") + '</td></tr>'
            '</table></div>'
        )
    else:
        html += ('<div class="gc" style="--gca:#27ae60;"><div class="gc-head">'
                 '<span class="gc-name">DDM H-Model</span></div>'
                 '<p style="color:var(--muted);font-size:13px;margin-top:12px;">'
                 'Requires positive dividend per share data.</p></div>')

    # 10. Mean Reversion
    mr = gr.get("mean_reversion")
    if mr:
        fv = mr["fair_value"]; ud_s, ud_c = ud(fv)
        fh, fc = gflag("Mean Reversion")
        n_comp = mr.get("n_components", 0)
        html += (
            '<div class="gc' + fc + '" style="--gca:#87ceeb;">'
            '<div class="gc-head"><span class="gc-name">Mean Reversion</span>'
            '<span class="gc-badge">Historical Context</span></div>' + fh +
            '<div class="gc-sub">Anchors valuation to each metric\'s own 5-year average multiple. '
            'Captures when a stock is cheap or expensive relative to its own history.</div>'
            '<div class="gc-fv">' + mo2(fv) + '</div>'
            '<div class="gc-ud ' + ud_c + '">' + ud_s + ' vs current price</div>'
            '<div class="gc-mos">MoS Price: ' + mo2(mr.get("mos_value")) + '</div><hr>'
            '<table class="gc-tbl">'
            '<tr><td>Components used</td><td>' + str(n_comp) + '</td></tr>' +
            ('<tr><td>P/E component</td><td>' + mo2(mr.get("pe_component")) + '</td></tr>' if mr.get("pe_component") else '') +
            ('<tr><td>P/FCF component</td><td>' + mo2(mr.get("pfcf_component")) + '</td></tr>' if mr.get("pfcf_component") else '') +
            ('<tr><td>EV/EBITDA component</td><td>' + mo2(mr.get("eveb_component")) + '</td></tr>' if mr.get("eveb_component") else '') +
            '</table></div>'
        )
    else:
        html += ('<div class="gc" style="--gca:#87ceeb;"><div class="gc-head">'
                 '<span class="gc-name">Mean Reversion</span></div>'
                 '<p style="color:var(--muted);font-size:13px;margin-top:12px;">'
                 'Requires 5-year historical P/E, P/FCF, or EV/EBITDA multiples.</p></div>')

    # 11. Bayesian Ensemble
    bayes = gr.get("bayesian")
    if bayes:
        fv = bayes["fair_value"]; ud_s, ud_c = ud(fv)
        n_m = bayes.get("n_models_used", 0)
        top3 = bayes.get("top_weighted_methods", [])
        top3_html = "".join(
            '<tr><td>' + t.get("method","") + '</td>'
            '<td>' + mo2(t.get("fv")) + '</td>'
            '<td>' + format(t.get("score",0), ".0f") + ' pts</td></tr>'
            for t in top3
        )
        surp_adj = bayes.get("surprise_adj_applied", False)
        html += (
            '<div class="gc" style="--gca:#ffd700;">'
            '<div class="gc-head"><span class="gc-name">Bayesian Ensemble</span>'
            '<span class="gc-badge">Meta · Consensus</span></div>'
            '<div class="gc-sub">Weighted average of all applicable models. Each model\'s weight equals '
            'its applicability score (data quality + company fit + academic precision).</div>'
            '<div class="gc-fv">' + mo2(fv) + '</div>'
            '<div class="gc-ud ' + ud_c + '">' + ud_s + ' vs current price</div>'
            '<div class="gc-mos">MoS Price: ' + mo2(bayes.get("mos_value")) + '</div><hr>'
            '<table class="gc-tbl">'
            '<tr><th>Top contributors</th><th>Fair Value</th><th>Weight</th></tr>' +
            top3_html +
            '<tr><td>Models in ensemble</td><td colspan="2">' + str(n_m) + '</td></tr>' +
            ('<tr><td>Earnings surprise adj</td><td colspan="2">Applied (±5%)</td></tr>' if surp_adj else '') +
            '</table></div>'
        )
    else:
        html += ('<div class="gc" style="--gca:#ffd700;"><div class="gc-head">'
                 '<span class="gc-name">Bayesian Ensemble</span></div>'
                 '<p style="color:var(--muted);font-size:13px;margin-top:12px;">'
                 'Requires at least 3 valid model results to compute weighted consensus.</p></div>')

    html+='</div>'  # close g-grid

    # ── Multi-method interpretation section ────────────────────────────────────
    html += _build_growth_interpretation(d, gr, reliability, price)

    return html

# ─────────────────────────────────────────────────────────────────
#  BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  HISTORICAL DATA — Stooq (prices) + SEC EDGAR (fundamentals)
#
#  Yahoo Finance has locked down their API and requires authentication
#  that is unreliable in automated contexts.
#
#  We use two fully open, no-key-required sources instead:
#
#  PRICES:       Stooq  (stooq.com) — plain CSV, no auth, wide coverage
#                Fallback: EODHD demo endpoint
#
#  FUNDAMENTALS: SEC EDGAR XBRL API (data.sec.gov) — official US govt API,
#                free, no key, covers every SEC-filing company (all US stocks)
# ─────────────────────────────────────────────────────────────────

import csv
import io
import http.cookiejar as cookiejar_mod

_REQ_HEADERS = {
    "User-Agent": "ValuationScript/1.0 research@example.com",   # SEC EDGAR requires a UA
    "Accept":     "*/*",
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    """Raw HTTP GET. Returns response bytes or b'' on failure."""
    try:
        req = urllib.request.Request(url, headers=_REQ_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return b""


# ── Price data via Stooq ──────────────────────────────────────────

def _stooq_symbol(ticker: str) -> str:
    """
    Convert a ticker to Stooq format.
    US equities: AAPL  → aapl.us
    Canadian:    SHOP  → shop.to   (if .TO suffix detected elsewhere; we try both)
    UK:          VOD   → vod.uk
    """
    return ticker.lower() + ".us"    # start with US; fallback tries others


def fetch_historical_prices(ticker: str, days: int, _unused=None) -> list:
    """
    Fetch daily closing prices using Stooq's free CSV endpoint.
    No authentication, no API key, no crumb.

    Falls back through several symbol variants if the first fails.
    Returns list of {"date": "YYYY-MM-DD", "close": float}, oldest-first,
    trimmed to the most recent `days` trading days.
    """
    # Stooq date range: go back 3× trading days to be safe (covers weekends + holidays)
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=int(days * 1.6) + 60)
    date_from = start_dt.strftime("%Y%m%d")
    date_to   = end_dt.strftime("%Y%m%d")

    suffixes = [".us", ".us", ".ca", ".uk", ".de", ""]   # .us twice: bare ticker retry
    t = ticker.lower()
    candidates = [t + s for s in suffixes]
    # Also try the ticker as-is (some indices / ETFs have no suffix on Stooq)
    candidates = list(dict.fromkeys(candidates))   # deduplicate, preserve order

    for sym in candidates:
        url = ("https://stooq.com/q/d/l/?s={sym}&d1={df}&d2={dt}&i=d"
               ).format(sym=sym, df=date_from, dt=date_to)
        raw = _http_get(url)
        if not raw or b"No data" in raw or b"Exceed" in raw or len(raw) < 50:
            continue

        try:
            text   = raw.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            prices = []
            for row in reader:
                try:
                    cl = float(row.get("Close") or row.get("close") or 0)
                    dt = (row.get("Date") or row.get("date") or "").strip()
                    if cl > 0 and dt:
                        # Stooq dates are YYYY-MM-DD already
                        prices.append({"date": dt, "close": cl})
                except (ValueError, TypeError):
                    continue
            if len(prices) < 10:
                continue
            prices.sort(key=lambda x: x["date"])
            print("  [Stooq] Got {} price points for symbol '{}'".format(len(prices), sym))
            return prices[-days:]
        except Exception:
            continue

    print("  [Stooq] All symbol variants failed for '{}'".format(ticker))
    return []


# ── Fundamental data via SEC EDGAR ───────────────────────────────

def _edgar_cik(ticker: str) -> str:
    """
    Look up the SEC CIK number for a ticker using EDGAR's company search API.
    Returns zero-padded 10-digit CIK string, or "" if not found.
    """
    url  = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2000-01-01&forms=10-K".format(
        urllib.parse.quote(ticker.upper()))
    # Preferred: use the ticker→CIK mapping file (much faster)
    map_url = "https://www.sec.gov/files/company_tickers.json"
    raw = _http_get(map_url, timeout=15)
    if raw:
        try:
            mapping = json.loads(raw)
            for entry in mapping.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    return cik
        except Exception:
            pass
    return ""


def _edgar_xbrl_facts(cik: str) -> dict:
    """
    Fetch the full XBRL company facts from SEC EDGAR for a given CIK.
    Returns the 'facts' sub-dict or {}.
    """
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json".format(cik)
    raw = _http_get(url, timeout=30)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data.get("facts", {})
    except Exception:
        return {}


def _xbrl_annual_series(facts: dict, *concept_candidates) -> list:
    """
    Extract annual (10-K) values for the first matching XBRL concept.
    Tries us-gaap namespace first, then dei.
    Returns list of {"date": "YYYY-MM-DD", "value": float} sorted oldest-first,
    deduplicated by fiscal year end date (keeping the most recent filing for each date).
    """
    for concept in concept_candidates:
        for ns in ["us-gaap", "dei"]:
            try:
                units = facts[ns][concept]["units"]
                # Revenue / income / cash concepts are in USD; shares in shares
                unit_key = "USD" if "USD" in units else list(units.keys())[0]
                entries  = units[unit_key]
                # Keep only 10-K annual filings, dedupe by period end
                annual = {}
                for e in entries:
                    if e.get("form") not in ("10-K", "10-K/A"):
                        continue
                    end  = e.get("end", "")
                    val  = e.get("val")
                    if end and val is not None:
                        # Keep the entry with the latest 'filed' date for each period end
                        existing = annual.get(end)
                        if existing is None or e.get("filed", "") > existing.get("filed", ""):
                            annual[end] = {"date": end, "value": float(val),
                                           "filed": e.get("filed", "")}
                if annual:
                    result = sorted(annual.values(), key=lambda x: x["date"])
                    return result
            except (KeyError, TypeError, IndexError):
                continue
    return []


def fetch_historical_fundamentals(ticker: str, _unused=None) -> list:
    """
    Fetch annual fundamental data from SEC EDGAR's free XBRL API.
    Requires no API key. Works for any SEC-filing company (all US-listed stocks).

    Returns a list of annual snapshots sorted oldest-first, each containing:
      {
        "date":        "YYYY-MM-DD",   # fiscal year end date
        "revenue":     float or None,
        "net_income":  float or None,
        "fcf":         float or None,  # operatingCashflow - capex (abs value)
        "total_debt":  float or None,
        "gross_margin":float or None,  # % (if available)
        "op_margin":   float or None,  # %
      }
    """
    print("  [EDGAR] Looking up CIK for {}...".format(ticker))
    cik = _edgar_cik(ticker)
    if not cik:
        print("  [EDGAR] CIK not found for '{}'".format(ticker))
        return []
    print("  [EDGAR] CIK: {}. Fetching XBRL facts...".format(cik))
    facts = _edgar_xbrl_facts(cik)
    if not facts:
        print("  [EDGAR] No XBRL facts returned for CIK {}".format(cik))
        return []

    # ── Pull each financial series ────────────────────────────────
    rev_series = _xbrl_annual_series(facts,
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "SalesRevenueGoodsNet", "RevenueFromContractWithCustomer")

    ni_series  = _xbrl_annual_series(facts,
        "NetIncomeLoss", "NetIncome", "ProfitLoss")

    opcf_series = _xbrl_annual_series(facts,
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")

    capex_series = _xbrl_annual_series(facts,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations",
        "PaymentsForCapitalImprovements")

    debt_series = _xbrl_annual_series(facts,
        "LongTermDebtNoncurrent", "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "DebtAndCapitalLeaseObligations")

    gp_series  = _xbrl_annual_series(facts, "GrossProfit")
    ebit_series= _xbrl_annual_series(facts,
        "OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes")

    def _as_dict(series):
        return {e["date"]: e["value"] for e in series}

    rev_d   = _as_dict(rev_series)
    ni_d    = _as_dict(ni_series)
    opcf_d  = _as_dict(opcf_series)
    capex_d = _as_dict(capex_series)
    debt_d  = _as_dict(debt_series)
    gp_d    = _as_dict(gp_series)
    ebit_d  = _as_dict(ebit_series)

    all_dates = sorted(set(list(rev_d.keys()) + list(ni_d.keys()) + list(opcf_d.keys())))
    if not all_dates:
        print("  [EDGAR] No usable annual series found")
        return []

    snapshots = []
    for date in all_dates:
        rev  = rev_d.get(date)
        ni   = ni_d.get(date)
        opcf = opcf_d.get(date)
        capex= capex_d.get(date)
        debt = debt_d.get(date)
        gp   = gp_d.get(date)
        ebit = ebit_d.get(date)

        # FCF = operating cash flow − capex (capex in XBRL is positive outflow)
        fcf = None
        if opcf is not None and capex is not None:
            fcf = opcf - capex
        elif opcf is not None:
            fcf = opcf * 0.85   # rough estimate

        gross_margin = (gp / rev * 100)   if (gp   and rev and rev > 0) else None
        op_margin    = (ebit / rev * 100) if (ebit and rev and rev > 0) else None

        snapshots.append({
            "date":        date,
            "revenue":     rev,
            "net_income":  ni,
            "fcf":         fcf,
            "total_debt":  debt,
            "gross_margin":gross_margin,
            "op_margin":   op_margin,
        })

    print("  [EDGAR] Found {} annual snapshots ({} → {})".format(
        len(snapshots), snapshots[0]["date"], snapshots[-1]["date"]))
    return snapshots


def _pick_snapshot_for_date(snapshots: list, target_date_str: str) -> dict:
    """
    Return the most recent snapshot whose filing date is <= target_date_str.
    Assumes annual 10-K filings are available approximately 75 days after fiscal year end.
    """
    target = target_date_str
    best = None
    for snap in snapshots:
        # A 10-K annual report is typically filed 60-90 days after fiscal year end
        period_end = snap["date"]
        try:
            avail_dt   = datetime.datetime.strptime(period_end, "%Y-%m-%d") + datetime.timedelta(days=75)
            avail_str  = avail_dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
        if avail_str <= target:
            best = snap
    return best


def _rebuild_d_from_snapshot(d_current: dict, snap: dict, hist_price: float,
                              benchmarks: dict) -> dict:
    """
    Build a historical 'd' data-dict by overlaying the snapshot's financials
    onto the current d-dict, replacing forward-looking fields with historical ones.
    Falls back to current values where historical data is unavailable.
    """
    # Start with a shallow copy of current d (preserves beta, sector, etc.)
    d_hist = dict(d_current)

    shares = d_current["shares"]  # use current shares as proxy

    def _or(new_val, fallback):
        """Use new_val if valid, else fallback."""
        if new_val is not None and new_val == new_val:  # not None, not NaN
            return new_val
        return fallback

    rev      = _or(snap.get("revenue"),    d_current["revenue"])
    ni       = _or(snap.get("net_income"), d_current["net_income"])
    fcf      = _or(snap.get("fcf"),        d_current["fcf"])
    debt     = _or(snap.get("total_debt"), d_current["total_debt"])
    gm       = _or(snap.get("gross_margin"), d_current["gross_margin"])
    op_mar   = _or(snap.get("op_margin"),  d_current["op_margin"])

    eps      = (ni / shares) if (ni and shares and shares > 0) else d_current["eps"]
    fcf_ps   = (fcf / shares) if (fcf and shares and shares > 0) else d_current["fcf_per_share"]
    fcf_mar  = (fcf / rev)    if (fcf and rev and rev > 0) else d_current["fcf_margin"]

    # Estimate EBITDA from op_margin or net_income (same logic as main fetch)
    ebitda = None
    ebitda_method = "N/A"
    if rev and op_mar:
        ebitda = rev * (op_mar / 100) + rev * 0.05
        ebitda_method = "operating income + est. D&A"
    elif ni:
        ebitda = ni * 1.35
        ebitda_method = "net income x 1.35"

    # Estimate growth from FCF margin (same logic as main fetch — no fwd data historically)
    if fcf_mar and fcf_mar > 0.40:   growth = 0.15
    elif fcf_mar and fcf_mar > 0.20: growth = 0.12
    elif fcf_mar and fcf_mar > 0.10: growth = 0.09
    else:                             growth = 0.06

    # Market cap and EV at the historical price
    mktcap      = hist_price * shares if shares else d_current["market_cap"]
    ev_approx   = mktcap + debt

    d_hist.update({
        "price":         hist_price,
        "market_cap":    mktcap,
        "revenue":       rev,
        "net_income":    ni,
        "fcf":           fcf,
        "fcf_per_share": fcf_ps,
        "fcf_margin":    fcf_mar,
        "total_debt":    debt,
        "cash":          d_current["cash"],   # cash not available historically; use current
        "ebitda":        ebitda,
        "ebitda_method": ebitda_method,
        "gross_margin":  gm,
        "op_margin":     op_mar,
        "eps":           eps,
        "shares":        shares,
        "est_growth":    growth,
        "growth_source": "FCF-margin proxy (historical)",
        "ev_approx":     ev_approx,
        "peg":           ((hist_price / eps) / (growth * 100)) if (eps and eps > 0 and hist_price) else None,
        "current_pe":    (hist_price / eps) if (eps and eps > 0) else None,
        "current_pfcf":  (mktcap / fcf) if (fcf and fcf > 0) else None,
        # Zero out forward-looking fields — not available historically
        "fwd_eps":       None,
        "fwd_rev":       None,
        "rev_growth_pct":None,
        "eps_growth_pct":None,
        "wacc_override": d_current.get("wacc_override"),
        "wacc_raw":      d_current.get("wacc_raw", {}),
    })
    return d_hist


def _run_methods_for_snapshot(d_hist: dict, benchmarks: dict, erg_peer_data: dict = None) -> dict:
    """
    Run all valuation methods on a historical d-dict.
    Returns {method_name: fair_value} for every method that produces a valid result.
    """
    results = {}
    bm = benchmarks

    if d_hist.get("fcf") and d_hist["fcf"] > 0 and d_hist.get("shares"):
        r = run_dcf(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["DCF"] = r["fair_value"]

    if d_hist.get("fcf_per_share") and d_hist["fcf_per_share"] > 0:
        r = run_pfcf(d_hist, bm)
        if r and r.get("fair_value", 0) > 0:
            results["P/FCF"] = r["fair_value"]

    if d_hist.get("eps") and d_hist["eps"] > 0:
        r = run_pe(d_hist, bm)
        if r and r.get("fair_value", 0) > 0:
            results["P/E"] = r["fair_value"]

    if d_hist.get("ebitda") and d_hist["ebitda"] > 0 and d_hist.get("shares"):
        r = run_ev_ebitda(d_hist, bm)
        if r and r.get("fair_value", 0) > 0:
            results["EV/EBITDA"] = r["fair_value"]

    r = run_reverse_dcf(d_hist)
    if r:
        for s in r.get("scenarios", []):
            if s["label"] == "Street" and s.get("fv", 0) > 0:
                results["Reverse DCF"] = s["fv"]
                break

    r = run_forward_peg(d_hist)
    if r and r.get("fair_value", 0) > 0:
        results["Fwd PEG"] = r["fair_value"]

    r = run_ev_ntm_revenue(d_hist)
    if r and r.get("fair_value", 0) > 0:
        results["EV/NTM Rev"] = r["fair_value"]

    r = run_tam_scenario(d_hist)
    if r and r.get("fair_value", 0) > 0:
        results["TAM Scenario"] = r["fair_value"]

    r = run_rule_of_40(d_hist)
    if r and r.get("fair_value", 0) > 0:
        results["Rule of 40"] = r["fair_value"]

    r = run_erg_valuation(d_hist, erg_peer_data)
    if r and r.get("fair_value", 0) > 0:
        results["ERG"] = r["fair_value"]

    if d_hist.get("fcf") and d_hist["fcf"] > 0 and d_hist.get("shares"):
        r = run_three_stage_dcf(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["Three-Stage DCF"] = r["fair_value"]

    if d_hist.get("fcf") and d_hist["fcf"] > 0 and d_hist.get("shares"):
        r = run_monte_carlo_dcf(d_hist, n_sims=500)
        if r and r.get("fair_value", 0) > 0:
            results["Monte Carlo DCF"] = r["fair_value"]

    if d_hist.get("fcf_per_share") and d_hist["fcf_per_share"] > 0:
        r = run_fcf_yield(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["FCF Yield"] = r["fair_value"]

    # Balance-sheet dependent models — use current ext data as proxy
    # (book value, roic, etc. change slowly year-on-year; a reasonable approximation)
    ext = d_hist.get("ext") or {}
    if ext.get("book_value_ps") and ext["book_value_ps"] > 0 and d_hist.get("eps", 0) > 0:
        r = run_rim(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["RIM"] = r["fair_value"]

    if ext.get("roic") and ext.get("stockholders_equity"):
        r = run_roic_excess_return(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["ROIC Excess Return"] = r["fair_value"]

    if ext.get("total_current_assets") is not None and ext.get("total_liabilities") is not None:
        r = run_ncav(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["NCAV"] = r["fair_value"]

    if ext.get("dividends_per_share") and ext["dividends_per_share"] > 0:
        r = run_ddm_hmodel(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["DDM"] = r["fair_value"]

    if d_hist.get("revenue") and d_hist.get("est_growth") and d_hist.get("shares"):
        r = run_scurve_tam(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["S-Curve TAM"] = r["fair_value"]

    if d_hist.get("price") and d_hist.get("revenue") and d_hist.get("shares"):
        r = run_pie(d_hist)
        if r and r.get("fair_value", 0) > 0:
            results["PIE"] = r["fair_value"]

    # Bayesian Ensemble — weighted consensus of all snapshot results
    if len(results) >= 3:
        _result_dicts = [{"method": m, "fair_value": fv, "mos_value": fv * 0.8}
                         for m, fv in results.items() if fv and fv > 0]
        _appl = {m: score_model_applicability(m, {"method": m, "fair_value": fv}, d_hist, [])
                 for m, fv in results.items() if fv and fv > 0}
        r = run_bayesian_ensemble(d_hist, _result_dicts, _appl, None)
        if r and r.get("fair_value", 0) > 0:
            results["Bayesian Ensemble"] = r["fair_value"]

    return results


def run_backtest(d: dict, results: list, gr: dict, days: int, erg_peer_data: dict = None) -> dict:
    """
    True time-series backtest.

    For every trading day in the backtest window:
      1. Determine which annual 10-K snapshot was publicly available on that date
         (annual reports are filed ~75 days after fiscal year end, so we apply that lag).
      2. Rebuild the fundamentals d-dict using that snapshot + the price on that date.
      3. Run all 9 valuation methods → one fair value per method per day.

    This produces evolving fair value curves (not flat lines) that respond to
    each annual earnings report as it was published.

    Scoring (MAPE across every trading day):
      MAPE = mean( |fv_t - price_t| / price_t ) * 100
      Accuracy = 100 - MAPE   (higher is better)

    Directional accuracy: did the method's BUY/SELL/HOLD signal on day-1 correctly
    predict whether the price rose or fell over the full window?
    """
    ticker = d["ticker"]

    print("[Backtest] Fetching {} trading days of price history from Stooq...".format(days))
    all_prices = fetch_historical_prices(ticker, days)
    if len(all_prices) < 10:
        return {"error": (
            "Could not retrieve sufficient historical price data for '{}' from Stooq "
            "(got {} data points). "
            "Check that the ticker is a valid US equity symbol."
        ).format(ticker, len(all_prices))}

    start_price      = all_prices[0]["close"]
    end_price        = all_prices[-1]["close"]
    start_date       = all_prices[0]["date"]
    end_date         = all_prices[-1]["date"]
    price_change_pct = (end_price - start_price) / start_price * 100

    print("[Backtest] Fetching annual fundamental history from SEC EDGAR...")
    snapshots = fetch_historical_fundamentals(ticker)

    if not snapshots:
        return {"error": (
            "Could not retrieve historical fundamental data for '{}' from SEC EDGAR. "
            "Only SEC-filing companies (US-listed stocks) are supported."
        ).format(ticker)}

    print("  Found {} annual snapshots ({} → {}).".format(
        len(snapshots), snapshots[0]["date"], snapshots[-1]["date"]))

    benchmarks_hist = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0}

    # ── Build a day-by-day time series of fair values ──────────────
    # For each trading day, find the most recent available snapshot (with 75-day lag)
    # and compute fair values. Cache by snapshot date to avoid redundant computation.
    fv_cache   = {}   # snap_date -> {method: fv}
    snap_cache = {}   # snap_date -> d_hist built from that snapshot

    # time_series[i] = {"date": ..., "price": ..., "fv": {method: float}, "snap_date": ...}
    time_series = []

    # Track how many unique snapshots were actually used
    snaps_used = set()

    for day in all_prices:
        date  = day["date"]
        price = day["close"]

        snap = _pick_snapshot_for_date(snapshots, date)
        if snap is None:
            # Before any filing is available — skip this day
            continue

        snap_date = snap["date"]
        snaps_used.add(snap_date)

        if snap_date not in fv_cache:
            d_h = _rebuild_d_from_snapshot(d, snap, price, benchmarks_hist)
            fvs = _run_methods_for_snapshot(d_h, benchmarks_hist, erg_peer_data)
            fv_cache[snap_date]   = fvs
            snap_cache[snap_date] = snap
        else:
            # Reuse cached fair values but update price-dependent metrics
            # (DCF, P/FCF, P/E etc. fair values don't depend on current price —
            # they're intrinsic values — so the cache is valid across days)
            fvs = fv_cache[snap_date]

        time_series.append({"date": date, "price": price,
                             "fv": fvs, "snap_date": snap_date})

    if not time_series:
        return {"error": "No trading days overlapped with available EDGAR filings for '{}'."
                         " Try a shorter backtest window or a different ticker.".format(ticker)}

    print("  Time series built: {} trading days, {} unique snapshots used.".format(
        len(time_series), len(snaps_used)))

    # ── Score each method across the full time series ──────────────
    all_methods = set()
    for row in time_series:
        all_methods.update(row["fv"].keys())

    method_stats = {}
    for name in all_methods:
        # Collect days where this method had a valid fair value
        daily_errors = []
        for row in time_series:
            fv = row["fv"].get(name)
            if fv and fv > 0:
                daily_errors.append(abs(fv - row["price"]) / row["price"] * 100)

        if len(daily_errors) < 5:
            continue

        mape     = sum(daily_errors) / len(daily_errors)
        accuracy = max(0.0, 100.0 - mape)

        # Directional: signal on first day vs actual outcome
        first_fv = None
        first_row_idx = None
        for i, row in enumerate(time_series):
            fv = row["fv"].get(name)
            if fv and fv > 0:
                first_fv = fv
                first_row_idx = i
                break

        if first_fv and first_fv > start_price * 1.05:
            signal = "BUY"
        elif first_fv and first_fv < start_price * 0.95:
            signal = "SELL"
        else:
            signal = "HOLD"

        # ── Directional accuracy: use a SHORT forward window (≤90 days) ──────
        # Measuring a BUY/SELL signal against a 1000-day return is meaningless —
        # no static valuation model is meant to predict multi-year price paths.
        # We use the first 90 trading days (or fewer if the window is shorter).
        DIR_WINDOW = min(90, len(time_series) - 1)
        if first_row_idx is not None and DIR_WINDOW > 0:
            end_idx = min(first_row_idx + DIR_WINDOW, len(time_series) - 1)
            fwd_price   = time_series[end_idx]["price"]
            fwd_chg_pct = (fwd_price - start_price) / start_price * 100
        else:
            fwd_chg_pct = price_change_pct  # fallback

        directional_correct = (
            (signal == "BUY"  and fwd_chg_pct > 2)  or
            (signal == "SELL" and fwd_chg_pct < -2) or
            (signal == "HOLD" and abs(fwd_chg_pct) <= 10)
        )

        # ── Point-in-time error: fv on day-1 vs price on day-1 ──────────────
        # This is the "was the model correctly valuing the stock on the start date?"
        # metric — totally separate from MAPE (which measures tracking over the full window).
        first_fv_val       = first_fv or 0
        pt_error           = abs(first_fv_val - start_price) / start_price * 100 if first_fv_val else 0

        # Upside implied by method on day 1 vs start price (correct comparison point)
        pred_upside        = (first_fv_val - start_price) / start_price * 100 if first_fv_val else 0
        pred_upside_str    = ("+" if pred_upside >= 0 else "") + format(pred_upside, ".1f") + "%"

        method_stats[name] = {
            "predicted_fv":        round(first_fv_val, 2),
            "mape":                round(mape, 1),            # mean error across all days
            "pct_error":           round(mape, 1),            # kept for backward compat
            "point_in_time_error": round(pt_error, 1),        # fv_day1 vs price_day1
            "accuracy_score":      round(accuracy, 1),        # 100 - MAPE
            "signal":              signal,
            "directional_correct": directional_correct,
            "dir_window_days":     DIR_WINDOW,                # so HTML can display it
            "fwd_chg_pct":         round(fwd_chg_pct, 1),    # actual return over dir window
            "predicted_upside":    round(pred_upside, 1),
            "pred_upside_str":     pred_upside_str,
            "coverage_days":       len(daily_errors),
        }

    if not method_stats:
        return {"error": "Scoring failed — no methods produced valid results for any trading day."}

    ranked       = sorted(method_stats.items(), key=lambda x: x[1]["accuracy_score"], reverse=True)
    best_method  = ranked[0][0]
    worst_method = ranked[-1][0]
    best_score   = ranked[0][1]["accuracy_score"]

    if best_score >= 85:
        overall_quality, overall_color = "EXCELLENT", "#00c896"
    elif best_score >= 70:
        overall_quality, overall_color = "GOOD",      "#4f8ef7"
    elif best_score >= 50:
        overall_quality, overall_color = "MODERATE",  "#f0a500"
    else:
        overall_quality, overall_color = "POOR",       "#e05c5c"

    # Snapshot used at the start of the window (for the HTML sidebar)
    start_snap = _pick_snapshot_for_date(snapshots, start_date)
    snap_date_display = start_snap["date"] if start_snap else snapshots[0]["date"]

    # ── Compute CURRENT prediction using live TradingView data ────
    # This is what the model predicts RIGHT NOW based on the latest
    # fundamentals — extends the chart into the present/future.
    current_fvs = _run_methods_for_snapshot(d, benchmarks_hist, erg_peer_data)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    return {
        "days":              days,
        "history":           all_prices,
        "time_series":       time_series,    # full day-by-day series for chart
        "start_date":        start_date,
        "end_date":          end_date,
        "start_price":       round(start_price, 2),
        "end_price":         round(end_price, 2),
        "price_change_pct":  round(price_change_pct, 1),
        "snap_date":         snap_date_display,
        "snaps_used":        sorted(snaps_used),
        "method_stats":      method_stats,
        "ranked":            ranked,
        "best_method":       best_method,
        "worst_method":      worst_method,
        "overall_quality":   overall_quality,
        "overall_color":     overall_color,
        "d_hist":            snap_cache.get(snap_date_display, {}),
        "current_fvs":       current_fvs,     # live prediction from current fundamentals
        "current_date":      current_date,
    }

def _build_backtest_html(bt: dict, d: dict, top_8: list = None, applicability_scores: dict = None) -> str:
    """Generate the HTML for the Backtesting tab."""
    if "error" in bt:
        return (
            '<div class="section-label">Backtesting — Historical Price Analysis</div>'
            '<div style="background:rgba(224,92,92,.1);border:1px solid rgba(224,92,92,.3);'
            'border-radius:10px;padding:28px 32px;color:#e89c9c;font-size:14px;">'
            '⚠ ' + bt["error"] + '</div>'
        )

    ticker      = d["ticker"]
    days        = bt["days"]
    history     = bt["history"]
    stats       = bt["method_stats"]
    ranked      = bt["ranked"]
    start_price = bt["start_price"]
    end_price   = bt["end_price"]
    chg_pct     = bt["price_change_pct"]
    start_date  = bt["start_date"]
    end_date    = bt["end_date"]
    snap_date   = bt["snap_date"]
    best        = bt["best_method"]
    worst       = bt["worst_method"]
    oq          = bt["overall_quality"]
    oc          = bt["overall_color"]
    d_hist      = bt.get("d_hist", {})

    chg_clr  = "#00c896" if chg_pct >= 0 else "#e05c5c"
    chg_str  = ("+" if chg_pct >= 0 else "") + format(chg_pct, ".1f") + "%"

    # ── SVG price chart — evolving fair value lines ────────────────
    # Use the full time_series (one entry per trading day) so fair value
    # lines step up/down each time a new annual filing becomes available.
    # SOLID lines = historical fair values (what the model said at each point)
    # DASHED lines = current prediction (what the model says NOW based on live data)
    time_series = bt.get("time_series", [])
    snaps_used  = bt.get("snaps_used", [])
    current_fvs = bt.get("current_fvs", {})
    current_date = bt.get("current_date", end_date)

    prices    = [row["price"] for row in time_series]
    ts_dates  = [row["date"]  for row in time_series]
    n         = len(prices)

    # We add extra space at the right for the current prediction extension
    # (approximately 10% of the chart width)
    PRED_EXTRA_FRAC = 0.10  # 10% extra width for prediction zone

    line_colors = {
        # Classic models
        "DCF":              "#4f8ef7",
        "Three-Stage DCF":  "#7baff7",
        "Monte Carlo DCF":  "#a8c8fb",
        "P/FCF":            "#00c896",
        "P/E":              "#f0a500",
        "EV/EBITDA":        "#bf6ff0",
        "FCF Yield":        "#34d399",
        "NCAV":             "#6ee7b7",
        "ROIC Excess Return": "#f472b6",
        "RIM":              "#fb923c",
        "Mean Reversion":   "#facc15",
        # Growth models
        "Reverse DCF":      "#e87c3e",
        "Fwd PEG":          "#f0d500",
        "EV/NTM Rev":       "#00e5c8",
        "TAM Scenario":     "#c45aff",
        "S-Curve TAM":      "#a855f7",
        "Rule of 40":       "#ff7aa2",
        "ERG":              "#38bdf8",
        "PIE":              "#ff6b9d",
        "DDM":              "#27ae60",
        # Meta models
        "Bayesian Ensemble":    "#f59e0b",
        "Multi-Factor":         "#10b981",
        # Legacy alias
        "Graham Number":    "#94a3b8",
    }

    # Determine which methods are in the top 8 for special rendering
    top_8_names = set()
    if top_8:
        for r in top_8:
            if isinstance(r, dict) and "method" in r:
                top_8_names.add(r["method"])

    # Y-axis range:
    # Include all FV values truthfully so lines sit at their correct dollar level.
    # Clip only truly extreme outliers (beyond 3× the price range above/below price)
    # so that e.g. a TAM Scenario at 10× price doesn't crush everything else.
    # The clipPath in the SVG will neatly cut any line that still goes out of bounds.
    if prices:
        price_lo = min(prices)
        price_hi = max(prices)
        price_span = max(price_hi - price_lo, price_hi * 0.10)

        # Collect all FV values (historical + current predictions)
        all_fv = []
        for row in time_series:
            for v in row["fv"].values():
                if v and v > 0:
                    all_fv.append(v)
        for v in current_fvs.values():
            if v and v > 0:
                all_fv.append(v)

        if all_fv:
            # Allow FV values up to 3× the price span above/below the price range
            hard_lo = price_lo - price_span * 3.0
            hard_hi = price_hi + price_span * 3.0
            fv_in_range = [v for v in all_fv if hard_lo <= v <= hard_hi]
            fv_lo = min(fv_in_range) if fv_in_range else price_lo
            fv_hi = max(fv_in_range) if fv_in_range else price_hi
        else:
            fv_lo, fv_hi = price_lo, price_hi

        p_min = min(price_lo, fv_lo) * 0.97
        p_max = max(price_hi, fv_hi) * 1.03
    else:
        p_min, p_max = 0, 1
    p_rng = p_max - p_min or 1
    # Canvas with inner padding so labels never overflow the SVG box
    W, H    = 860, 420   # total SVG canvas
    PAD_L   = 52         # left  — room for Y-axis labels
    PAD_R   = 12         # right  — small breathing room
    PAD_T   = 28         # top   — room for filing date labels
    PAD_B   = 8          # bottom
    PW      = W - PAD_L - PAD_R   # plot width
    PH      = H - PAD_T - PAD_B   # plot height

    # Historical data occupies (1 - PRED_EXTRA_FRAC) of the plot width
    HIST_W  = PW * (1 - PRED_EXTRA_FRAC)   # width for historical data
    PRED_X  = PAD_L + HIST_W                # x-coordinate where prediction zone starts

    def to_svg_x(i):
        """Map historical index to SVG x (within the historical zone)."""
        return round(PAD_L + i / max(n - 1, 1) * HIST_W, 1)

    def to_svg_y(val):
        val = max(p_min, min(p_max, val))
        return round(PAD_T + PH - (val - p_min) / p_rng * PH, 1)

    # Price polyline (historical only)
    price_pts = " ".join("{},{}".format(to_svg_x(i), to_svg_y(p))
                         for i, p in enumerate(prices))

    # Per-method fair value polylines (solid = historical, dashed = current prediction)
    # Each method's lines are wrapped in a <g id="btline-METHOD_ID"> so JS can toggle them.
    fv_polylines  = ""
    method_final_y = {}   # mname -> final Y coordinate (for legend ordering top→bottom)

    def _line_id(mname):
        """Sanitise a method name into a valid HTML/SVG element id."""
        return "btline-" + mname.lower().replace("/", "-").replace(" ", "-").replace("_", "-")

    # Prediction zone background (subtle shading)
    pred_zone_bg = (
        '<rect x="{px}" y="{pt}" width="{pw}" height="{ph}" '
        'fill="rgba(79,142,247,0.04)" />'
        '<line x1="{px}" y1="{pt}" x2="{px}" y2="{pb}" '
        'stroke="#333" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>'
        '<text x="{tx}" y="{ty}" fill="#555" font-size="8" text-anchor="middle" '
        'font-family="monospace">CURRENT</text>'
    ).format(
        px=round(PRED_X, 1), pt=PAD_T, pw=round(PW * PRED_EXTRA_FRAC, 1),
        ph=PH, pb=H-PAD_B,
        tx=round(PRED_X + PW * PRED_EXTRA_FRAC / 2, 1), ty=PAD_T - 4,
    )

    # X position for the current prediction endpoint (right edge of prediction zone)
    pred_end_x = round(PAD_L + PW - 4, 1)

    for mname in sorted(stats.keys()):
        clr      = line_colors.get(mname, "#aaaaaa")
        is_best  = (mname == best)
        is_top8  = (mname in top_8_names)
        sw       = "2.5" if is_best else ("2.0" if is_top8 else "1.5")
        op       = "1.0" if is_best else ("0.85" if is_top8 else "0.55")
        gid      = _line_id(mname)

        # Historical line: SOLID
        pts_list = []
        last_fv  = None
        last_x   = None
        for i, row in enumerate(time_series):
            fv = row["fv"].get(mname)
            if fv and fv > 0:
                x = to_svg_x(i)
                pts_list.append("{},{}".format(x, to_svg_y(fv)))
                last_fv = fv
                last_x  = x

        if len(pts_list) < 2:
            continue

        # Build this method's SVG content
        method_svg = (
            '<polyline points="{pts}" fill="none" stroke="{c}" '
            'stroke-width="{sw}" stroke-linejoin="round" opacity="{op}"/>'
        ).format(pts=" ".join(pts_list), c=clr, sw=sw, op=op)

        # Draw dashed prediction extension from last historical point to current prediction
        cur_fv = current_fvs.get(mname)
        if cur_fv and cur_fv > 0 and last_fv is not None and last_x is not None:
            pred_y = to_svg_y(cur_fv)
            last_hist_y = to_svg_y(last_fv)
            method_svg += (
                '<polyline points="{x0},{y0} {x1},{y1} {x2},{y2}" fill="none" stroke="{c}" '
                'stroke-width="{sw}" stroke-linejoin="round" stroke-dasharray="6,3" opacity="{op}"/>'
            ).format(
                x0=last_x, y0=last_hist_y,
                x1=round(PRED_X, 1), y1=last_hist_y,
                x2=pred_end_x, y2=pred_y,
                c=clr, sw=sw, op=op,
            )
            method_final_y[mname] = pred_y
        elif last_fv is not None:
            method_final_y[mname] = to_svg_y(last_fv)

        # All lines start hidden — user toggles them on via the cards below the chart
        fv_polylines += '<g id="{gid}" style="display:none">{svg}</g>'.format(
            gid=gid, svg=method_svg
        )

    # Order legend top-to-bottom by final Y value (smallest Y = highest on chart = drawn on top)
    # Price line (white) goes first in legend
    legend_order = sorted(method_final_y.keys(), key=lambda m: method_final_y[m])
    fv_legend = (
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'margin:3px 10px 3px 0;font-size:11px;color:#ffffff;font-weight:600;">'
        '<span style="display:inline-block;width:18px;height:2px;background:#ffffff;border-radius:2px;"></span>'
        'Price</span>'
    )
    for mname in legend_order:
        clr      = line_colors.get(mname, "#aaaaaa")
        is_best  = (mname == best)
        is_top8  = (mname in top_8_names)
        lc       = "#ffffff" if is_best else ("#e2e8f0" if is_top8 else "#888")
        fw       = "700" if is_best else ("600" if is_top8 else "400")
        star     = " ★" if is_best else ""
        top8_dot = (
            ' <span style="font-size:8px;vertical-align:middle;color:#4f8ef7;">●</span>'
        ) if is_top8 and not is_best else ""
        line_h   = "2.5px" if is_top8 else "1.5px"
        fv_legend += (
            '<span style="display:inline-flex;align-items:center;gap:5px;'
            'margin:3px 10px 3px 0;font-size:11px;color:{lc};font-weight:{fw};">'
            '<span style="display:inline-block;width:18px;height:{lh};'
            'background:{c};border-radius:2px;"></span>{n}{best}{dot}</span>'
        ).format(c=clr, lh=line_h, n=mname, lc=lc, fw=fw, best=star, dot=top8_dot)

    # Add solid/dashed key + top-8 indicator key to legend
    fv_legend += (
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'margin:3px 10px 3px 0;font-size:10px;color:#777;">'
        '( ── historical &nbsp; - - - current prediction &nbsp;·&nbsp; '
        '<span style="color:#4f8ef7;">●</span> = top-8 method )</span>'
    )

    # Vertical tick marks at each filing date boundary (labels inside top padding)
    filing_ticks = ""
    for snap_d in snaps_used:
        for i, row in enumerate(time_series):
            if row["snap_date"] == snap_d:
                x = to_svg_x(i)
                filing_ticks += (
                    '<line x1="{x}" y1="{pt}" x2="{x}" y2="{pb}" '
                    'stroke="#444" stroke-width="1" stroke-dasharray="2,4" opacity="0.6"/>' +
                    '<text x="{x}" y="{ty}" fill="#666" font-size="8" text-anchor="middle" '
                    'font-family="monospace">{d}</text>'
                ).format(x=x, pt=PAD_T, pb=H-PAD_B, ty=PAD_T-4, d=snap_d)
                break

    # Y-axis gridlines and labels (inside the plot area, labels in left padding)
    y_gridlines = ""
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        val = p_min + frac * p_rng
        y   = to_svg_y(val)
        y_gridlines += (
            '<line x1="{pl}" y1="{y}" x2="{pr}" y2="{y}" '
            'stroke="#1e2030" stroke-width="1"/>' +
            '<text x="{tx}" y="{ty}" fill="#666" font-size="9" text-anchor="end" '
            'font-family="monospace">${v:.0f}</text>'
        ).format(pl=PAD_L, pr=W-PAD_R, y=y,
                 tx=PAD_L-4, ty=round(y+3,1), v=val)

    svg_chart = (
        '<svg viewBox="0 0 {W} {H}" style="width:100%;height:{H}px;display:block;"' +
        ' xmlns="http://www.w3.org/2000/svg">' +
        '<defs><clipPath id="plot-area">'
        '<rect x="{pl}" y="{pt}" width="{pw}" height="{ph}"/></clipPath></defs>' +
        '<rect width="{W}" height="{H}" fill="#0e1117" rx="6"/>' +
        y_gridlines +
        filing_ticks +
        pred_zone_bg +
        '<g clip-path="url(#plot-area)">' +
        fv_polylines +
        '<polyline points="{pts}" fill="none" stroke="#ffffff" ' +
        'stroke-width="2" stroke-linejoin="round" opacity="0.95"/>' +
        '</g>' +
        '</svg>'
    ).format(W=W, H=H, pl=PAD_L, pt=PAD_T, pw=PW, ph=PH, pts=price_pts)

    # ── Fundamentals used table ─────────────────────────────────────
    def _fmt(v, prefix="$", suffix="", billions=False, pct=False):
        if v is None:
            return "N/A"
        if pct:
            return format(v, ".1f") + "%"
        if billions:
            return prefix + format(abs(v) / 1e9, ",.2f") + "B"
        return prefix + format(v, ",.2f") + suffix

    hist_fund_rows = ""
    fund_fields = [
        ("TTM Revenue",    _fmt(d_hist.get("revenue"),    billions=True)),
        ("TTM Net Income", _fmt(d_hist.get("net_income"), billions=True)),
        ("TTM FCF",        _fmt(d_hist.get("fcf"),        billions=True)),
        ("EPS (TTM)",      _fmt(d_hist.get("eps"))),
        ("Gross Margin",   _fmt(d_hist.get("gross_margin"), prefix="", suffix="", pct=True)),
        ("Op. Margin",     _fmt(d_hist.get("op_margin"),    prefix="", suffix="", pct=True)),
        ("Total Debt",     _fmt(d_hist.get("total_debt"),  billions=True)),
        ("Est. Growth",    _fmt(d_hist.get("est_growth", 0) * 100, prefix="", suffix="%")),
        ("Price at Start", "$" + format(start_price, ",.2f")),
        ("Snapshot Date",  snap_date + " (period end)"),
    ]
    for k, v in fund_fields:
        hist_fund_rows += (
            '<div style="display:flex;justify-content:space-between;padding:7px 0;'
            'border-bottom:1px solid var(--border);font-size:13px;">'
            '<span style="color:var(--muted);">{k}</span>'
            '<span style="font-family:\'DM Mono\',monospace;">{v}</span>'
            '</div>'
        ).format(k=k, v=v)

    # ── Results table rows ──────────────────────────────────────────
    rows_html = ""
    medals = ["🥇", "🥈", "🥉"]
    for rank_i, (mname, mst) in enumerate(ranked):
        medal    = medals[rank_i] if rank_i < 3 else "&nbsp;#" + str(rank_i + 1)
        score    = mst["accuracy_score"]
        err      = mst["mape"]
        bar_w    = max(2, score)
        bar_clr  = "#00c896" if score >= 75 else "#f0a500" if score >= 50 else "#e05c5c"
        sig      = mst["signal"]
        sig_clr  = {"BUY": "#00c896", "SELL": "#e05c5c", "HOLD": "#f0a500"}.get(sig, "#aaa")
        dir_str  = "✓ Correct"   if mst["directional_correct"] else "✗ Incorrect"
        dir_clr  = "#00c896"     if mst["directional_correct"] else "#e05c5c"
        pred_upside = mst["pred_upside_str"]          # fv_day1 vs start_price
        pu_clr   = "#00c896" if mst["predicted_upside"] >= 0 else "#e05c5c"
        pt_err   = mst["point_in_time_error"]         # |fv_day1 - start_price| / start_price
        pt_clr   = "#00c896" if pt_err <= 10 else "#f0a500" if pt_err <= 25 else "#e05c5c"
        is_best  = (mname == best)
        is_top8  = (mname in top_8_names)
        row_bg   = "rgba(79,142,247,.06)" if is_best else ("rgba(79,142,247,.02)" if is_top8 else "")

        best_tag = ""
        if is_best:
            best_tag = (' <span style="font-size:9px;background:rgba(79,142,247,.2);'
                        'color:#4f8ef7;padding:2px 6px;border-radius:3px;margin-left:6px;'
                        'font-family:\'DM Mono\',monospace;letter-spacing:1px;">BEST</span>')
        elif is_top8:
            best_tag = (' <span style="font-size:9px;background:rgba(79,142,247,.1);'
                        'color:#4f8ef7;padding:2px 6px;border-radius:3px;margin-left:6px;'
                        'font-family:\'DM Mono\',monospace;letter-spacing:1px;">TOP 8</span>')

        rows_html += """
        <tr style="background:{rbg};">
          <td style="font-weight:700;font-size:16px;width:44px;text-align:center;padding:12px 6px;">{medal}</td>
          <td style="font-weight:700;color:var(--text);padding:12px 10px;">{mname}{best_tag}</td>
          <td style="text-align:center;font-family:'DM Mono',monospace;padding:12px 10px;">${fv:.2f}</td>
          <td style="text-align:center;color:{pu_clr};font-family:'DM Mono',monospace;padding:12px 10px;">{pred_upside} <span style="font-size:10px;color:#666;">({pt_err:.1f}% off)</span></td>
          <td style="padding:12px 10px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="flex:1;background:var(--surface2);border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{bw:.0f}%;height:8px;background:{bc};border-radius:4px;transition:width .3s;"></div>
              </div>
              <span style="font-family:'DM Mono',monospace;font-size:12px;min-width:42px;text-align:right;">{score:.1f}</span>
            </div>
          </td>
          <td style="text-align:center;font-family:'DM Mono',monospace;color:#aaa;padding:12px 10px;">{err:.1f}%</td>
          <td style="text-align:center;color:{sc};font-size:12px;font-weight:700;padding:12px 10px;">{sig}</td>
          <td style="text-align:center;color:{dc};font-size:12px;padding:12px 10px;">{dir} <span style="font-size:10px;color:#666;">(+{dw}d)</span></td>
        </tr>""".format(
            rbg=row_bg, medal=medal, mname=mname, best_tag=best_tag,
            fv=mst["predicted_fv"], pu_clr=pu_clr, pred_upside=pred_upside,
            pt_err=pt_err,
            bw=bar_w, bc=bar_clr, score=score, err=err,
            sc=sig_clr, sig=sig, dc=dir_clr, dir=dir_str,
            dw=mst.get("dir_window_days", 90),
        )

    # ── Verdict text ────────────────────────────────────────────────
    best_stat  = stats[best]
    best_score = best_stat["accuracy_score"]
    best_mape  = best_stat["mape"]
    best_fv    = best_stat["predicted_fv"]
    best_pt    = best_stat["point_in_time_error"]
    dir_window = best_stat.get("dir_window_days", 90)

    if best_score >= 85:
        verdict_detail = (
            "The <strong>{best}</strong> method was the most accurate tracker of {ticker}'s price "
            "over this {days}-day window, with a mean tracking error (MAPE) of just "
            "<strong>{mape:.1f}%</strong> across all {days} trading days. "
            "On the start date ({sd}), it estimated a fair value of <strong>${fv:.2f}</strong> "
            "against an actual price of <strong>${sp:.2f}</strong> — a point-in-time error of "
            "<strong>{pt:.1f}%</strong>. "
            "This suggests {best} captures how the market prices {ticker} better than any other "
            "method in this backtest window."
        ).format(best=best, ticker=ticker, days=days, mape=best_mape, sd=start_date,
                 fv=best_fv, sp=start_price, pt=best_pt)
    elif best_score >= 65:
        verdict_detail = (
            "<strong>{best}</strong> had the tightest tracking of {ticker}'s actual price over "
            "this {days}-day window (MAPE: <strong>{mape:.1f}%</strong>). "
            "On the start date ({sd}), it estimated a fair value of <strong>${fv:.2f}</strong> "
            "against an actual price of <strong>${sp:.2f}</strong> ({pt:.1f}% off). "
            "While not a perfect fit, it outperformed the other methods. "
            "Consider combining it with the 2nd-ranked method for a more robust estimate."
        ).format(best=best, ticker=ticker, days=days, mape=best_mape, sd=start_date,
                 fv=best_fv, sp=start_price, pt=best_pt)
    else:
        verdict_detail = (
            "No method tracked {ticker}'s price closely over this {days}-day window. "
            "The best performer (<strong>{best}</strong>) had a mean tracking error (MAPE) of "
            "<strong>{mape:.1f}%</strong> across the window. On the start date ({sd}), "
            "it estimated a fair value of <strong>${fv:.2f}</strong> against an actual price of "
            "<strong>${sp:.2f}</strong> ({pt:.1f}% off on that day). "
            "This suggests {ticker} may be driven more by sentiment, macro conditions, or sector "
            "rotation than by the fundamental metrics these models rely on — or that the backtest "
            "window spans a period of unusually large fundamental re-rating."
        ).format(best=best, ticker=ticker, days=days, mape=best_mape, sd=start_date,
                 fv=best_fv, sp=start_price, pt=best_pt)

    dir_hits  = sum(1 for s in stats.values() if s["directional_correct"])
    dir_total = len(stats)

    # Build clickable current prediction mini-cards (toggle chart lines on/off)
    # All lines start hidden. Top-8 cards appear first (with TOP 8 badge) then other backtest methods.
    def _make_card(mname, fv, clr, has_backtest_line):
        """Return a toggle card HTML string for a method."""
        gid        = _line_id(mname)
        is_top8    = mname in top_8_names
        upside_pct = (fv - d["price"]) / d["price"] * 100
        up_clr     = "#00c896" if upside_pct >= 0 else "#e05c5c"
        up_str     = ("+" if upside_pct >= 0 else "") + format(upside_pct, ".0f") + "%"
        top8_badge = (
            ' <span style="font-size:8px;background:rgba(79,142,247,.2);color:#4f8ef7;'
            'padding:1px 5px;border-radius:3px;letter-spacing:1px;font-weight:700;">TOP 8</span>'
        ) if is_top8 else ""
        if has_backtest_line:
            onclick_attr  = 'onclick="btToggleLine(\'{gid}\')"'.format(gid=gid)
            cursor_style  = "cursor:pointer;"
            title_attr    = 'title="Click to show/hide this line on the chart"'
            eye_html      = '<div id="eye-{gid}" style="font-size:12px;opacity:0.7;">🙈</div>'.format(gid=gid)
        else:
            onclick_attr  = 'onclick="btNoLine(\'{mn}\')"'.format(mn=mname.replace("'", "\\'"))
            cursor_style  = "cursor:default;"
            title_attr    = 'title="No historical chart line — this model uses current-period data not available historically"'
            eye_html      = '<div style="font-size:10px;opacity:0.4;">—</div>'
        return (
            '<div id="card-{gid}" '
            '{onclick} '
            'data-visible="false" '
            'style="background:var(--surface2);border-radius:8px;padding:10px 14px;'
            'min-width:120px;border-left:3px solid {clr};{cursor}'
            'transition:opacity .2s,box-shadow .2s;user-select:none;opacity:0.35;" '
            '{title}>'
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:4px;gap:8px;">'
            '<div style="font-size:11px;color:var(--muted);">{mname}{badge}</div>'
            '{eye}'
            '</div>'
            '<div style="font-size:16px;font-weight:700;font-family:\'DM Mono\',monospace;">'
            '${fv:.2f}</div>'
            '<div style="font-size:11px;color:{up_clr};font-family:\'DM Mono\',monospace;">'
            '{up_str} vs ${price:.2f}</div>'
            '</div>'
        ).format(
            gid=gid, onclick=onclick_attr, clr=clr, cursor=cursor_style, title=title_attr,
            mname=mname, badge=top8_badge, eye=eye_html,
            fv=fv, up_clr=up_clr, up_str=up_str, price=d["price"]
        )

    def _resolve_fv(mname):
        """Get the best available fair value for a method."""
        fv = current_fvs.get(mname)
        if not fv or fv <= 0:
            for r2 in (top_8 or []):
                if r2.get("method") == mname and r2.get("fair_value", 0) > 0:
                    return r2["fair_value"]
        return fv

    # ── Top 12 by backtest accuracy — same ranking as All Methods tab ─
    # Primary sort: methods tracked in backtest, by accuracy_score descending.
    # Secondary: methods with no backtest line, by applicability score.
    seen_methods = set()
    all_available = []

    for mname, mst in sorted(stats.items(), key=lambda x: x[1]["accuracy_score"], reverse=True):
        fv = _resolve_fv(mname)
        if fv and fv > 0 and mname not in seen_methods:
            all_available.append((mname, fv))
            seen_methods.add(mname)

    # Supplement with top_8 models that have no backtest line (e.g. ext-data models)
    for r2 in (top_8 or []):
        mname = r2.get("method")
        if mname not in seen_methods:
            fv = _resolve_fv(mname)
            if fv and fv > 0:
                all_available.append((mname, fv))
                seen_methods.add(mname)

    top12_cards = [
        _make_card(mname, fv, line_colors.get(mname, "#aaaaaa"), mname in method_final_y)
        for mname, fv in all_available[:12]
    ]

    current_pred_cards_html = ""
    if top12_cards:
        current_pred_cards_html += (
            '<div style="width:100%;font-size:10px;font-family:\'DM Mono\',monospace;'
            'letter-spacing:2px;text-transform:uppercase;color:#4f8ef7;'
            'margin-bottom:6px;">Top 12 Methods — Ranked by Backtest Tracking Accuracy</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:12px;">'
            + "".join(top12_cards) + "</div>"
        )

    # ── Assemble HTML ───────────────────────────────────────────────
    html = (
        '<div class="section-label">Backtesting — True {days}-Day Historical Prediction Test</div>'
        '<div class="g-note">'
        '<strong>Methodology:</strong>&nbsp; For every trading day in this {days}-day window, '
        'the script identifies which annual 10-K filing was publicly available on that date '
        '(applying a 75-day filing lag) and re-runs all valuation methods using those '
        'historical fundamentals. This produces <strong>evolving fair value curves</strong> that '
        'step up or down each time a new annual report was filed. '
        '<strong>Tracking Accuracy</strong> = 100 − MAPE, where MAPE = mean(|fv − price| / price × 100) '
        'across every trading day — this measures how closely the model\'s fair value tracked the actual price. '
        '<strong>Point-in-time error</strong> (shown in the "vs Start Price" column) measures how far each '
        'model\'s day-1 estimate was from the actual price on that same day — a separate question. '
        '<strong>Directional signal accuracy</strong> is measured over the first 90 trading days only '
        '(not the full window), since a static valuation model\'s BUY/SELL signal cannot be expected '
        'to predict multi-year price paths. '
        'Vertical dashed marks on the chart show when a new 10-K filing became available. '
        'The method with the highest tracking accuracy is ranked #1.'
        '</div>'

        # KPI row
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));'
        'gap:16px;margin-bottom:28px;">'
        '<div class="range-item"><span class="range-label">Price on {sd}</span>'
        '<span class="range-value" style="color:var(--text);font-size:22px;">${sp:.2f}</span></div>'
        '<div class="range-item"><span class="range-label">Price Today</span>'
        '<span class="range-value" style="color:var(--text);font-size:22px;">${ep:.2f}</span></div>'
        '<div class="range-item"><span class="range-label">{days}-Day Return</span>'
        '<span class="range-value" style="color:{cc};font-size:22px;">{cs}</span></div>'
        '<div class="range-item"><span class="range-label">Best Predictor</span>'
        '<span class="range-value" style="color:#4f8ef7;font-size:16px;">{best}</span></div>'
        '<div class="range-item"><span class="range-label">Directional Hits</span>'
        '<span class="range-value" style="color:{oc};font-size:22px;">{dh}/{dt}</span></div>'
        '</div>'

        # Price chart
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;'
        'padding:16px 20px;margin-bottom:8px;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:var(--muted);margin-bottom:10px;">'
        '{ticker} price: {sd} → {ed} &nbsp;·&nbsp; solid = historical fair values &nbsp;·&nbsp; dashed = current prediction</div>'
        '{chart}'
        '<div style="margin-top:10px;line-height:1.8;">{legend}</div>'
        '</div>'

        # Current Predictions summary
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;'
        'padding:16px 20px;margin-bottom:20px;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:var(--accent);margin-bottom:12px;">'
        'Current Fair Value Predictions (Live Data)</div>'
        '{current_pred_cards}'
        '<div style="font-size:11px;color:var(--muted);margin-top:10px;">'
        '&#128065; Click a card to show or hide its line on the chart above.'
        '</div>'
        '</div>'

        # Two-column layout: results table + historical fundamentals
        '<div style="display:grid;grid-template-columns:1fr 280px;gap:20px;margin-bottom:32px;'
        'align-items:start;">'

        # Left: results table
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:var(--surface2);">'
        '<th style="padding:11px 6px;text-align:center;width:44px;">#</th>'
        '<th style="padding:11px 10px;text-align:left;">Method</th>'
        '<th style="padding:11px 10px;text-align:center;">FV on Day 1</th>'
        '<th style="padding:11px 10px;text-align:center;">vs Start Price</th>'
        '<th style="padding:11px 10px;text-align:left;min-width:180px;">Tracking Accuracy (MAPE)</th>'
        '<th style="padding:11px 10px;text-align:center;">Avg Tracking Error</th>'
        '<th style="padding:11px 10px;text-align:center;">Day-1 Signal</th>'
        '<th style="padding:11px 10px;text-align:center;">Signal Correct? (90d)</th>'
        '</tr></thead><tbody>{rows}</tbody></table>'
        '<div style="padding:10px 16px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);">'
        '<strong style="color:var(--text);">FV on Day 1</strong> = fair value computed using fundamentals available on the backtest start date. '
        '<strong style="color:var(--text);">vs Start Price</strong> = how far that estimate was from the actual price on the same day (point-in-time accuracy). '
        '<strong style="color:var(--text);">Tracking Accuracy</strong> = 100 − MAPE, where MAPE = mean(|fv − price| / price) across every trading day in the window. '
        '<strong style="color:var(--text);">Signal Correct?</strong> = did the BUY/SELL/HOLD signal on day 1 correctly predict the direction of price over the next 90 trading days.</div>'
        '</div>'

        # Right: historical fundamentals used
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:2px;'
        'text-transform:uppercase;color:var(--accent);margin-bottom:14px;">'
        'Fundamentals Used</div>'
        '<div style="font-size:11px;color:var(--muted);margin-bottom:12px;line-height:1.5;">'
        'From Yahoo Finance quarterly filing for period ending <strong style="color:var(--text);">{snap}</strong>. '
        'TTM = trailing-12-month sum of 4 quarterly periods.'
        '</div>'
        '{fund_rows}'
        '</div>'
        '</div>'

        # Verdict banner
        '<div style="background:var(--surface);border:1px solid var(--border);'
        'border-left:4px solid {oc};border-radius:12px;padding:28px 32px;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:3px;'
        'text-transform:uppercase;color:var(--muted);margin-bottom:12px;">Backtesting Verdict</div>'
        '<div style="font-size:26px;font-weight:800;color:{oc};margin-bottom:12px;">'
        '{oq} — Best Predictor: {best}</div>'
        '<p style="font-size:14px;color:var(--muted);line-height:1.7;max-width:800px;">'
        '{vd}</p>'
        '<div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);'
        'font-size:12px;color:var(--muted);line-height:1.7;">'
        '<strong style="color:var(--text);">⚠ Limitations:</strong>&nbsp;'
        'Historical annual fundamentals are sourced from SEC EDGAR\'s free XBRL API (data.sec.gov). '
        'Historical prices are sourced from Stooq (stooq.com). '
        'Shares outstanding and cash are proxied from current data (not available historically). '
        'Growth rates in the historical snapshots are estimated from FCF margin only — '
        'no analyst forward estimates are available for past dates, so methods that rely heavily on '
        'forward EPS/revenue may be less accurate historically than they would be with real consensus data. '
        'The directional signal is evaluated over a 90-trading-day forward window — '
        'not the full backtest window — so it measures short-term signal quality, not long-term price prediction. '
        'Use this as directional evidence, not proof of predictive power.'
        '</div></div>'
    ).format(
        days=days, sd=start_date, snap=snap_date, ed=end_date,
        sp=start_price, ep=end_price, cc=chg_clr, cs=chg_str,
        best=best, dh=dir_hits, dt=dir_total, oc=oc,
        ticker=ticker, chart=svg_chart, legend=fv_legend,
        rows=rows_html, fund_rows=hist_fund_rows,
        oq=oq, vd=verdict_detail,
        current_pred_cards=current_pred_cards_html,
    )
    return html


def _build_all_methods_table(top_8: list, remainder: list, applicability_scores: dict, method_colors: dict, price: float, bt: dict = None) -> str:
    """Build the ranked 'All Methods' table HTML."""
    CAT = {
        "DCF": "Classic", "P/FCF": "Classic", "P/E": "Classic",
        "EV/EBITDA": "Classic", "Three-Stage DCF": "Classic",
        "Monte Carlo DCF": "Classic", "FCF Yield": "Classic",
        "RIM": "Classic", "ROIC Excess Return": "Classic",
        "NCAV": "Classic", "Graham Number": "Classic",
        "Reverse DCF": "Growth", "Fwd PEG": "Growth",
        "EV/NTM Rev": "Growth", "TAM Scenario": "Growth",
        "Rule of 40": "Growth", "ERG": "Growth",
        "DDM": "Growth", "S-Curve TAM": "Growth",
        "PIE": "Growth", "Mean Reversion": "Growth",
        "Bayesian Ensemble": "Meta", "Multi-Factor": "Meta",
    }
    CAT_CSS = {"Classic": "cat-classic", "Growth": "cat-growth", "Meta": "cat-meta"}

    # Extract backtest method stats if available
    bt_stats = {}
    if bt and "error" not in bt and "method_stats" in bt:
        bt_stats = bt["method_stats"]
    has_bt = bool(bt_stats)

    all_models = list(top_8) + [r for r in (remainder or []) if r.get("fair_value") and r["fair_value"] > 0]
    none_models = [r for r in (remainder or []) if not r.get("fair_value") or not r["fair_value"]]

    # When backtest data is available, re-rank all models by tracking accuracy.
    # Models with no backtest data sort below those that do (using applicability score as tiebreak).
    if has_bt:
        def _rank_key(r):
            m = r.get("method", "")
            if m in bt_stats:
                return (1, bt_stats[m]["accuracy_score"])   # higher = better
            return (0, applicability_scores.get(m, 0))      # no backtest → below ranked ones
        all_models.sort(key=_rank_key, reverse=True)

    rows_html = ""
    rank = 0
    top_8_methods = {r.get("method") for r in top_8}

    for r in all_models:
        rank += 1
        method = r.get("method", "")
        fv = r.get("fair_value")
        mos = r.get("mos_value")
        appl_score = applicability_scores.get(method, 0)
        cat = CAT.get(method, "Classic")
        cat_css = CAT_CSS.get(cat, "cat-classic")
        col = method_colors.get(method, "#aaa")
        is_top8 = method in top_8_methods
        row_style = ' style="background:rgba(79,142,247,.03);"' if is_top8 else ""

        if fv and price and price > 0:
            upside = (fv - price) / price * 100
            upside_str = ("+" if upside >= 0 else "") + "{:.1f}%".format(upside)
            upside_col = "#00c896" if upside >= 0 else "#e05c5c"
        else:
            upside_str = "N/A"
            upside_col = "#6b7194"

        fv_str  = "${:,.2f}".format(fv)  if fv  else "N/A"
        mos_str = "${:,.2f}".format(mos) if mos else "N/A"
        top8_badge = (
            ' <span style="font-size:9px;background:rgba(79,142,247,.2);color:#4f8ef7;'
            'padding:1px 5px;border-radius:3px;">TOP 8</span>'
        ) if is_top8 else ""

        # Backtest accuracy cell
        if has_bt and method in bt_stats:
            bt_acc  = bt_stats[method]["accuracy_score"]
            bt_mape = bt_stats[method]["mape"]
            bt_clr  = "#00c896" if bt_acc >= 75 else "#f0a500" if bt_acc >= 50 else "#e05c5c"
            bt_cell = (
                "<td style='font-family:DM Mono,monospace;color:{c};text-align:center;'>"
                "{acc:.0f} <span style='font-size:10px;color:#666;'>({mape:.0f}% err)</span>"
                "</td>"
            ).format(c=bt_clr, acc=bt_acc, mape=bt_mape)
        elif has_bt:
            bt_cell = "<td style='color:var(--muted);text-align:center;font-size:11px;'>No data</td>"
        else:
            bt_cell = ""

        score_cell = (
            "<td class='score-col' style='color:#888;'>{score:.0f}/100</td>"
        ).format(score=appl_score)

        rows_html += (
            "<tr{row_style}>"
            "<td class='rank-col'>#{rank}</td>"
            "<td><span style='color:{col};font-weight:700;'>{method}</span>{top8_badge}</td>"
            "<td><span class='cat-badge {cat_css}'>{cat}</span></td>"
            "<td style='font-family:DM Mono,monospace;'>{fv_str}</td>"
            "<td style='font-family:DM Mono,monospace;color:{upside_col};'>{upside_str}</td>"
            "<td style='font-family:DM Mono,monospace;'>{mos_str}</td>"
            "{bt_cell}"
            "{score_cell}"
            "</tr>"
        ).format(
            row_style=row_style, rank=rank, col=col, method=method,
            top8_badge=top8_badge, cat_css=cat_css, cat=cat,
            fv_str=fv_str, upside_col=upside_col, upside_str=upside_str,
            mos_str=mos_str, bt_cell=bt_cell, score_cell=score_cell,
        )

    # N/A section
    na_colspan = "8" if has_bt else "7"
    if none_models:
        rows_html += (
            "<tr><td colspan='{c}' style='padding:12px 10px;color:var(--muted);"
            "font-size:11px;border-top:2px solid var(--border);'>"
            "Not Applicable (insufficient data for this stock)</td></tr>"
        ).format(c=na_colspan)
        for r in none_models:
            method = r.get("method", "")
            cat = CAT.get(method, "Classic")
            cat_css = CAT_CSS.get(cat, "cat-classic")
            col = method_colors.get(method, "#aaa")
            reason = r.get("skip_reason", "Insufficient data or conditions not met")
            rows_html += (
                "<tr class='na-row'>"
                "<td class='rank-col'>—</td>"
                "<td><span style='color:{col};'>{method}</span></td>"
                "<td><span class='cat-badge {cat_css}'>{cat}</span></td>"
                "<td colspan='{c}' style='color:var(--muted);'>{reason}</td>"
                "</tr>"
            ).format(col=col, method=method, cat_css=cat_css, cat=cat,
                     reason=reason, c=str(int(na_colspan) - 3))

    bt_header = "<th style='text-align:center;'>Backtest Accuracy</th>" if has_bt else ""
    rank_note = (
        " <span style='font-size:10px;font-weight:400;color:var(--muted);'>"
        "(ranked by backtest tracking accuracy)</span>"
        if has_bt else
        " <span style='font-size:10px;font-weight:400;color:var(--muted);'>"
        "(ranked by applicability score — run with --backtest to rank by historical accuracy)</span>"
    )

    return (
        "<div style='font-size:12px;color:var(--muted);margin-bottom:10px;line-height:1.6;'>"
        + ("Ranked by how closely each model tracked {}'s actual price over the backtest window. "
           "Models with no historical data are listed below those with backtest results.".format("") if has_bt else
           "Run with <code>--backtest DAYS</code> to rank by historical price-tracking accuracy.")
        + "</div>"
        "<table class='all-methods-tbl'>"
        "<thead><tr>"
        "<th>Rank{note}</th><th>Method</th><th>Category</th>"
        "<th>Fair Value</th><th>Upside%</th><th>MoS Price</th>"
        "{bt_header}"
        "<th>Appl. Score</th>"
        "</tr></thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table>"
    ).format(note=rank_note, bt_header=bt_header)


def generate_html(d: dict, results: list, conv: dict, benchmarks: dict, gr: dict = None, reliability: dict = None, bt: dict = None, applicability_scores: dict = None, top_8: list = None, remainder: list = None) -> str:
    gr = gr or {}
    reliability = reliability or {}
    price        = d["price"]
    ticker       = d["ticker"]
    name         = d["name"]
    company_name = d.get("company_name", ticker)
    ts      = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    applicability_scores = applicability_scores or {}
    top_8    = top_8    or results
    remainder = remainder or []

    method_colors = {
        "DCF":               "#4f8ef7",
        "P/FCF":             "#00c896",
        "P/E":               "#f0a500",
        "EV/EBITDA":         "#bf6ff0",
        "Three-Stage DCF":   "#3ec9f5",
        "Monte Carlo DCF":   "#7b68ee",
        "FCF Yield":         "#20b2aa",
        "RIM":               "#ff8c69",
        "ROIC Excess Return":"#98fb98",
        "DDM":               "#27ae60",
        "NCAV":              "#daa520",
        "S-Curve TAM":       "#c45aff",
        "PIE":               "#ff6b9d",
        "Mean Reversion":    "#87ceeb",
        "Bayesian Ensemble": "#ffd700",
        "Multi-Factor":      "#ff7f50",
        "Fwd PEG":           "#f0d500",
        "EV/NTM Rev":        "#00e5c8",
        "TAM Scenario":      "#c45aff",
        "Rule of 40":        "#ff7aa2",
        "ERG":               "#38bdf8",
        "Reverse DCF":       "#e87c3e",
        "Graham Number":     "#cd853f",
    }

    method_icons = {
        "DCF":               "⟳",
        "P/FCF":             "₣",
        "P/E":               "₱",
        "EV/EBITDA":         "Ξ",
        "Three-Stage DCF":   "③",
        "Monte Carlo DCF":   "🎲",
        "FCF Yield":         "⌀",
        "RIM":               "ℝ",
        "ROIC Excess Return":"◉",
        "DDM":               "∂",
        "NCAV":              "₦",
        "S-Curve TAM":       "∫",
        "PIE":               "π",
        "Mean Reversion":    "↩",
        "Bayesian Ensemble": "∑",
        "Multi-Factor":      "✦",
        "Fwd PEG":           "↗",
        "EV/NTM Rev":        "∈",
        "TAM Scenario":      "◻",
        "Rule of 40":        "40",
        "ERG":               "℮",
        "Reverse DCF":       "↙",
        "Graham Number":     "G",
    }

    # Build method cards (top 8 only)
    method_cards_html = ""
    display_list = top_8 if top_8 else results
    for r in display_list:
        fv    = r["fair_value"]
        mos   = r["mos_value"]
        label, color_cls = ud(fv, price)
        col   = method_colors.get(r["method"], "#4f8ef7")
        icon  = method_icons.get(r["method"], "◆")
        is_conv = r in conv["converging"]
        badge = '<span class="badge converging">CONVERGING</span>' if is_conv else '<span class="badge diverging">OUTLIER</span>'

        sub_rows = ""
        meth = r["method"]
        if meth in ("DCF", "Three-Stage DCF"):
            sub_rows = (
                "<tr><td>Growth Rate</td><td>" + pc(r.get("growth")) + "</td></tr>"
                "<tr><td>WACC (" + r.get("wacc_source","CAPM") + ")</td><td>" + pc(r.get("wacc")) + "</td></tr>"
                "<tr><td>Terminal Value %</td><td>" + (format(r["details"]["term_pct"], ".0f") + "%" if r.get("details") else "N/A") + "</td></tr>"
                "<tr><td>Net Debt</td><td>" + B(r.get("net_debt")) + "</td></tr>"
            )
            if r.get("warning"):
                sub_rows += "<tr><td colspan=\"2\" style=\"color:#f0a500;font-size:11px;padding-top:8px;\">" + r["warning"] + "</td></tr>"
        elif meth == "Monte Carlo DCF":
            sub_rows = (
                "<tr><td>Simulations</td><td>" + str(r.get("n_sims", 5000)) + "</td></tr>"
                "<tr><td>P10 / P25</td><td>" + mo(r.get("p10")) + " / " + mo(r.get("p25")) + "</td></tr>"
                "<tr><td>P75 / P90</td><td>" + mo(r.get("p75")) + " / " + mo(r.get("p90")) + "</td></tr>"
                "<tr><td>Convergence</td><td>" + (format(r.get("convergence_pct", 0), ".0f") + "%") + "</td></tr>"
                "<tr><td>Std Dev</td><td>" + mo(r.get("std_dev")) + "</td></tr>"
            )
        elif meth == "FCF Yield":
            sub_rows = (
                "<tr><td>FCF per Share</td><td>" + mo(r.get("fcf_per_share")) + "</td></tr>"
                "<tr><td>Target Yield</td><td>" + pc(r.get("target_yield")) + "</td></tr>"
                "<tr><td>Current Yield</td><td>" + pc(r.get("current_yield")) + "</td></tr>"
                "<tr><td>Yield Spread</td><td>" + pc(r.get("yield_spread")) + "</td></tr>"
            )
        elif meth == "RIM":
            sub_rows = (
                "<tr><td>Book Value/Share</td><td>" + mo(r.get("book_value_ps")) + "</td></tr>"
                "<tr><td>Cost of Equity (Ke)</td><td>" + pc(r.get("ke")) + "</td></tr>"
                "<tr><td>Payout Ratio</td><td>" + pc(r.get("payout_ratio")) + "</td></tr>"
                "<tr><td>PV Residual Income</td><td>" + mo(r.get("pv_ri")) + "</td></tr>"
            )
        elif meth == "ROIC Excess Return":
            sub_rows = (
                "<tr><td>ROIC</td><td>" + pc(r.get("roic")) + "</td></tr>"
                "<tr><td>WACC</td><td>" + pc(r.get("wacc")) + "</td></tr>"
                "<tr><td>IC per Share</td><td>" + mo(r.get("ic_ps")) + "</td></tr>"
                "<tr><td>ROIC spread</td><td>" + pc(r.get("roic_spread")) + "</td></tr>"
            )
            if r.get("warning"):
                sub_rows += "<tr><td colspan=\"2\" style=\"color:#f0a500;font-size:11px;\">" + r["warning"] + "</td></tr>"
        elif meth == "NCAV":
            sub_rows = (
                "<tr><td>NCAV per Share</td><td>" + mo(r.get("ncav_ps")) + "</td></tr>"
                "<tr><td>Graham Verdict</td><td>" + str(r.get("graham_verdict", "N/A")) + "</td></tr>"
            )
        elif meth == "Mean Reversion":
            sub_rows = (
                "<tr><td>Components Used</td><td>" + str(r.get("n_components", 0)) + "</td></tr>"
            )
            for comp in r.get("components", []):
                sub_rows += "<tr><td>" + str(comp.get("name","")) + "</td><td>" + mo(comp.get("value")) + "</td></tr>"
        elif meth == "Bayesian Ensemble":
            sub_rows = (
                "<tr><td>Models Used</td><td>" + str(r.get("n_models_used", 0)) + "</td></tr>"
                "<tr><td>Weighted FV</td><td>" + mo(r.get("weighted_fv")) + "</td></tr>"
            )
            if r.get("earnings_adj"):
                sub_rows += "<tr><td>Earnings Adj</td><td>" + r["earnings_adj"] + "</td></tr>"
        elif meth == "Multi-Factor":
            sub_rows = (
                "<tr><td>Total Score</td><td>" + str(r.get("total_score", 0)) + "/100</td></tr>"
                "<tr><td>Expected Return</td><td>" + pc(r.get("expected_return")) + "</td></tr>"
                "<tr><td>Value Score</td><td>" + str(r.get("value_score", 0)) + "/25</td></tr>"
                "<tr><td>Quality Score</td><td>" + str(r.get("quality_score", 0)) + "/25</td></tr>"
                "<tr><td>Growth Score</td><td>" + str(r.get("growth_score", 0)) + "/25</td></tr>"
                "<tr><td>Momentum Score</td><td>" + str(r.get("momentum_score", 0)) + "/25</td></tr>"
            )
        elif meth in ("P/FCF", "P/E", "EV/EBITDA") and r.get("fv_target") is not None:
            # Classic multiples-based methods with fv_target/fv_market/fv_conserv
            if meth == "P/FCF":
                t_label = "Target (" + format(r["target_multiple"], ".0f") + "x PEG-adj)"
                c_label = "Conservative (" + format(r["conserv_multiple"], ".0f") + "x)"
            elif meth == "P/E":
                t_label = "Target (" + format(r.get("target_mult", 25), ".0f") + "x PEG-adj)"
                c_label = "Conservative (" + format(r.get("conserv_mult", 15), ".0f") + "x)"
            else:
                t_label = "Target (" + format(r.get("target_mult", 20), ".0f") + "x growth-adj)"
                c_label = "Conservative (" + format(r.get("conserv_mult", 10), ".0f") + "x)"
            sub_rows = (
                "<tr><td>" + t_label + "</td><td>" + mo(r["fv_target"]) + "</td></tr>"
                "<tr><td>At " + r.get("peer_label", "Sector Median") + "</td><td>" + mo(r["fv_market"]) + "</td></tr>"
                "<tr><td>" + c_label + "</td><td>" + mo(r["fv_conserv"]) + "</td></tr>"
            )
            if meth == "P/E" and r.get("peg"):
                sub_rows += "<tr><td>PEG Ratio</td><td>" + format(r["peg"], ".2f") + "x</td></tr>"
            if meth == "EV/EBITDA":
                sub_rows += "<tr><td>EBITDA</td><td>" + B(r.get("ebitda")) + "</td></tr>"
        else:
            # Generic fallback — show any numeric keys
            skip_keys = {"method","fair_value","mos_value","reliable","warning",
                         "debt_heavy","skip_reason","details","scenarios","components"}
            for k, v in r.items():
                if k in skip_keys:
                    continue
                if isinstance(v, float):
                    sub_rows += "<tr><td>" + k.replace("_"," ").title() + "</td><td>" + format(v, ",.2f") + "</td></tr>"
                elif isinstance(v, int):
                    sub_rows += "<tr><td>" + k.replace("_"," ").title() + "</td><td>" + str(v) + "</td></tr>"
            if r.get("warning"):
                sub_rows += "<tr><td colspan=\"2\" style=\"color:#f0a500;font-size:11px;\">" + r["warning"] + "</td></tr>"

        # Build reliability flag HTML for this method
        rel = reliability.get(r["method"], {})
        rel_flags = rel.get("flags", [])
        flag_html = ""
        if rel_flags:
            flag_items = "".join('<div class="flag-item">⚠ ' + f + '</div>' for f in rel_flags)
            flag_html = '<div class="reliability-flag">' + flag_items + '</div>'

        appl_score = applicability_scores.get(r["method"], 0)
        method_cards_html += """
        <div class="method-card {flagged_cls}" style="--accent: {col};">
            <div class="method-header">
                <span class="method-icon">{icon}</span>
                <span class="method-name">{method}</span>
                {badge}
                {flag_badge}
            </div>
            {flag_html}
            <div class="method-fv">{fv}</div>
            <div class="method-updown {color_cls}">{label} vs current price</div>
            <div class="method-mos">MoS Price: {mos}</div>
            <table class="method-detail">
                {sub_rows}
            </table>
            <div class="appl-bar-wrap">
                <div class="appl-bar-track"><div class="appl-bar-fill" style="width:{appl_pct}%;background:{col};"></div></div>
                <span class="appl-score-lbl">{appl_score:.0f}/100 applicability</span>
            </div>
        </div>
        """.format(
            col=col, icon=icon, method=r["method"], badge=badge,
            fv=mo(fv), color_cls=color_cls, label=label,
            mos=mo(mos), sub_rows=sub_rows,
            flag_html=flag_html,
            flag_badge='<span class="badge flag-badge">LOW CONFIDENCE</span>' if rel_flags else '',
            flagged_cls="method-flagged" if rel_flags else '',
            appl_score=appl_score,
            appl_pct=min(100, appl_score),
        )

    # Convergence bars — position each method on a price spectrum
    display_list_fvs = top_8 if top_8 else results
    all_fvs_with_price = [price] + [r["fair_value"] for r in display_list_fvs if r.get("fair_value")]
    bar_min  = min(all_fvs_with_price) * 0.85
    bar_max  = max(all_fvs_with_price) * 1.15
    bar_range = bar_max - bar_min

    def bar_pct(val):
        return round((val - bar_min) / bar_range * 100, 1)

    conv_bars = ""
    for r in display_list_fvs:
        fv  = r.get("fair_value")
        if not fv:
            continue
        pct_pos = bar_pct(fv)
        col = method_colors.get(r["method"], "#aaa")
        conv_bars += """
        <div class="bar-row">
            <span class="bar-label">{method}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct}%; background:{col};"></div>
                <span class="bar-val">{val}</span>
            </div>
        </div>
        """.format(method=r["method"], pct=pct_pos, col=col, val=mo(fv))

    price_pct = bar_pct(price)
    conv_bars += """
        <div class="bar-row bar-price-row">
            <span class="bar-label">◆ Current Price</span>
            <div class="bar-track">
                <div class="bar-fill price-bar" style="width:{pct}%;"></div>
                <span class="bar-val">{val}</span>
            </div>
        </div>
    """.format(pct=price_pct, val=mo(price))

    # Divergence analysis section
    div_html = ""
    if conv["diverging"]:
        for r in conv["diverging"]:
            diff = abs(r["fair_value"] - conv["median"]) / conv["median"] * 100
            direction = "above" if r["fair_value"] > conv["median"] else "below"
            interp = ""
            if r["method"] == "DCF" and r["fair_value"] > conv["median"]:
                interp = "The DCF is producing a higher estimate — check if the growth rate assumption is too optimistic, or if terminal value is dominating (>80% of EV is a red flag)."
            elif r["method"] == "DCF" and r["fair_value"] < conv["median"]:
                interp = "The DCF is more pessimistic than multiples-based methods. Consider whether the discount rate (WACC) is too high or growth projections too conservative."
            elif r["method"] == "EV/EBITDA" and r["fair_value"] > conv["median"]:
                interp = "EV/EBITDA implies a higher value — this often occurs when a company has high EBITDA margins but lower reported earnings due to D&A or interest costs."
            elif r["method"] == "EV/EBITDA" and r["fair_value"] < conv["median"]:
                interp = "EV/EBITDA is the most conservative here — the company may carry more debt than peers, inflating EV and compressing the implied equity value."
            elif r["method"] == "P/E" and r["fair_value"] > conv["median"]:
                interp = "P/E implies more upside — earnings may be temporarily depressed (e.g. high D&A, one-time charges), making the stock look expensive on earnings but reasonable on cash flow."
            elif r["method"] == "P/E" and r["fair_value"] < conv["median"]:
                interp = "P/E is more bearish — this can occur when earnings are elevated by one-time items or buybacks. FCF-based methods may be more representative of true earning power."
            elif r["method"] == "P/FCF" and r["fair_value"] > conv["median"]:
                interp = "P/FCF implies the highest value — the company converts earnings to cash exceptionally well. High FCF relative to net income often signals conservative accounting."
            elif r["method"] == "P/FCF" and r["fair_value"] < conv["median"]:
                interp = "P/FCF is the most conservative — high capex or working capital needs may be suppressing FCF. Dig into the cash flow statement to understand the gap."
            else:
                interp = "This method diverges from the consensus. Review the underlying inputs for this method against the company's latest filings."

            div_html += """
            <div class="divergence-item">
                <div class="div-header">
                    <span class="div-method" style="color:{col};">{method}</span>
                    <span class="div-diff">{diff:.1f}% {direction} consensus</span>
                </div>
                <p class="div-interp">{interp}</p>
            </div>
            """.format(
                col=method_colors[r["method"]],
                method=r["method"],
                diff=diff,
                direction=direction,
                interp=interp
            )
    else:
        div_html = '<p class="no-divergence">All top methods are within 15% of each other — strong convergence signal.</p>'

    # Verdict color
    verdict_colors = {
        "UNDERVALUED":  "#00c896",
        "OVERVALUED":   "#e05c5c",
        "FAIRLY VALUED":"#f0a500",
    }
    verdict_color = verdict_colors.get(conv["verdict"], "#f0a500")
    upside_sign = "+" if conv["upside"] >= 0 else ""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — Valuation Report</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
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
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    padding: 0 0 80px;
  }}

  /* ── HEADER ── */
  .hero {{
    background: linear-gradient(135deg, #0d0f14 0%, #111827 50%, #0d0f14 100%);
    border-bottom: 1px solid var(--border);
    padding: 60px 48px 48px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(79,142,247,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .hero-label {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }}
  .hero-title {{
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 60px);
    line-height: 1.05;
    color: var(--text);
    margin-bottom: 8px;
  }}
  .hero-subtitle {{
    font-size: 15px;
    color: var(--muted);
    margin-bottom: 36px;
  }}
  .hero-meta {{
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
  }}
  .meta-item {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .meta-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .meta-value {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }}
  .meta-value.verdict {{
    color: {verdict_color};
  }}

  /* ── LAYOUT ── */
  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 48px;
  }}
  section {{
    margin-top: 56px;
  }}
  .section-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}

  /* ── CONVICTION BANNER ── */
  .conviction-banner {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid {conviction_color};
    border-radius: 12px;
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 32px;
    flex-wrap: wrap;
    margin-top: 40px;
  }}
  .conviction-score {{
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    color: {conviction_color};
    line-height: 1;
  }}
  .conviction-info h2 {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .conviction-info p {{
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
  }}
  .conviction-stats {{
    margin-left: auto;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
  }}
  .cstat {{
    text-align: right;
  }}
  .cstat-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .cstat-value {{
    font-size: 20px;
    font-weight: 700;
    margin-top: 2px;
  }}

  /* ── METHOD CARDS ── */
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
  }}
  .method-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    padding: 24px;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .method-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
  }}
  .method-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }}
  .method-icon {{
    font-size: 20px;
    color: var(--accent);
  }}
  .method-name {{
    font-weight: 700;
    font-size: 15px;
    flex: 1;
  }}
  .badge {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
  }}
  .badge.converging {{
    background: rgba(0,200,150,0.15);
    color: var(--up);
    border: 1px solid rgba(0,200,150,0.3);
  }}
  .badge.diverging {{
    background: rgba(240,165,0,0.15);
    color: var(--warn);
    border: 1px solid rgba(240,165,0,0.3);
  }}
  .badge.flag-badge {{
    background: rgba(224,92,92,0.15);
    color: var(--down);
    border: 1px solid rgba(224,92,92,0.3);
    margin-left: 4px;
  }}
  .reliability-flag {{
    background: rgba(224,92,92,0.08);
    border: 1px solid rgba(224,92,92,0.25);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 8px 0 4px;
  }}
  .flag-item {{
    font-size: 11px;
    color: #e89c9c;
    line-height: 1.5;
    margin-bottom: 2px;
  }}
  .flag-item:last-child {{ margin-bottom: 0; }}
  .method-flagged {{
    opacity: 0.70;
    border: 1px dashed rgba(224,92,92,0.4) !important;
  }}
  .method-flagged:hover {{
    opacity: 1.0;
  }}
  .method-fv {{
    font-family: 'DM Serif Display', serif;
    font-size: 38px;
    line-height: 1;
    margin-bottom: 6px;
    color: var(--accent);
  }}
  .method-updown {{
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 4px;
  }}
  .method-updown.up   {{ color: var(--up); }}
  .method-updown.down {{ color: var(--down); }}
  .method-updown.neutral {{ color: var(--muted); }}
  .method-mos {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 18px;
    font-family: 'DM Mono', monospace;
  }}
  .method-detail {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .method-detail td {{
    padding: 5px 0;
    border-top: 1px solid var(--border);
    color: var(--muted);
  }}
  .method-detail td:last-child {{
    text-align: right;
    color: var(--text);
    font-family: 'DM Mono', monospace;
  }}

  /* ── CONVERGENCE BARS ── */
  .bar-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
  }}
  .bar-label {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    width: 100px;
    flex-shrink: 0;
    text-align: right;
  }}
  .bar-track {{
    flex: 1;
    background: var(--surface2);
    border-radius: 4px;
    height: 28px;
    position: relative;
    overflow: visible;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    min-width: 4px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
  }}
  .bar-fill.price-bar {{
    background: rgba(255,255,255,0.15) !important;
    border: 2px dashed rgba(255,255,255,0.4);
  }}
  .bar-price-row .bar-label {{
    color: var(--text);
    font-weight: 600;
  }}
  .bar-val {{
    position: absolute;
    right: -72px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
  }}
  .bar-wrapper {{
    padding-right: 80px;
  }}

  /* ── RANGE SUMMARY ── */
  .range-summary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 24px;
  }}
  .range-item {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .range-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .range-value {{
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
  }}
  .range-value.green  {{ color: var(--up); }}
  .range-value.red    {{ color: var(--down); }}
  .range-value.yellow {{ color: var(--warn); }}
  .range-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }}

  /* ── DIVERGENCE ── */
  .divergence-item {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 14px;
  }}
  .div-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
  }}
  .div-method {{
    font-weight: 700;
    font-size: 15px;
  }}
  .div-diff {{
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--warn);
    background: rgba(240,165,0,0.1);
    padding: 3px 10px;
    border-radius: 4px;
  }}
  .div-interp {{
    font-size: 14px;
    color: var(--muted);
    line-height: 1.7;
  }}
  .no-divergence {{
    color: var(--up);
    font-size: 15px;
    padding: 20px;
    background: rgba(0,200,150,0.08);
    border: 1px solid rgba(0,200,150,0.2);
    border-radius: 10px;
  }}

  /* ── FUNDAMENTALS TABLE ── */
  .fundamentals {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 700px) {{
    .fundamentals {{ grid-template-columns: 1fr; }}
    .hero {{ padding: 40px 24px 36px; }}
    .container {{ padding: 0 24px; }}
  }}
  .fund-table {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  .fund-table-title {{
    padding: 14px 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    color: var(--muted);
  }}
  .fund-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 11px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }}
  .fund-row:last-child {{ border-bottom: none; }}
  .fund-key {{ color: var(--muted); }}
  .fund-val {{ font-family: 'DM Mono', monospace; font-weight: 500; }}

  /* ── FOOTER ── */
  .footer {{
    margin-top: 72px;
    padding: 32px 48px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.8;
  }}

  /* ══ TABS ═══════════════════════════════════════════════════════ */
  .tab-bar{{display:flex;border-bottom:2px solid var(--border);background:var(--bg);position:sticky;top:0;z-index:20;}}
  .tab-btn{{background:none;border:none;border-bottom:3px solid transparent;margin-bottom:-2px;padding:16px 32px;color:var(--muted);font-family:'DM Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:color .15s,border-color .15s;}}
  .tab-btn:hover{{color:var(--text);}}
  .tab-btn.active{{color:var(--accent);border-bottom-color:var(--accent);}}
  .tab-panel{{display:none;}}
  .tab-panel.active{{display:block;}}

  /* ══ GROWTH CARDS ════════════════════════════════════════════════ */
  .g-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;margin-top:28px;}}
  .gc{{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--gca,var(--accent));border-radius:12px;padding:22px 24px 20px;}}
  .gc-head{{display:flex;align-items:center;gap:10px;margin-bottom:4px;}}
  .gc-name{{font-weight:700;font-size:14px;flex:1;}}
  .gc-badge{{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;padding:3px 8px;border-radius:4px;background:rgba(79,142,247,.1);color:var(--accent);border:1px solid rgba(79,142,247,.25);}}
  .gc-sub{{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:14px;}}
  .gc-fv{{font-family:'DM Serif Display',serif;font-size:34px;line-height:1;color:var(--gca,var(--accent));margin-bottom:3px;}}
  .gc-ud{{font-family:'DM Mono',monospace;font-size:12px;margin-bottom:3px;}}
  .gc-ud.up{{color:var(--up);}} .gc-ud.down{{color:var(--down);}}
  .gc-mos{{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);margin-bottom:14px;}}
  .gc hr{{border:none;border-top:1px solid var(--border);margin:10px 0;}}
  .gc-tbl{{width:100%;border-collapse:collapse;font-size:12px;}}
  .gc-tbl td{{padding:5px 0;border-top:1px solid var(--border);color:var(--muted);vertical-align:top;}}
  .gc-tbl td:last-child{{text-align:right;color:var(--text);font-family:'DM Mono',monospace;}}
  .gc-tbl tr:first-child td{{border-top:none;}}
  .sc-tbl{{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px;}}
  .sc-tbl th{{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:4px 6px;text-align:left;border-bottom:1px solid var(--border);}}
  .sc-tbl th:not(:first-child){{text-align:right;}}
  .sc-tbl td{{padding:5px 6px;border-top:1px solid var(--border);font-family:'DM Mono',monospace;font-size:11px;}}
  .sc-tbl td:not(:first-child){{text-align:right;}}
  .sc-tbl .hi{{background:rgba(79,142,247,.07);}}
  .sup{{color:var(--up);}} .sdn{{color:var(--down);}}
  .ig-box{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 18px;display:flex;align-items:center;gap:16px;margin-bottom:14px;}}
  .ig-num{{font-family:'DM Serif Display',serif;font-size:36px;color:var(--warn);flex-shrink:0;line-height:1;}}
  .ig-lbl{{font-size:12px;color:var(--muted);line-height:1.5;}}
  .ig-vs{{margin-left:auto;text-align:right;}}
  .ig-vs-k{{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);}}
  .ig-vs-v{{font-family:'DM Mono',monospace;font-size:13px;font-weight:600;color:var(--text);}}
  .ro40-num{{font-family:'DM Serif Display',serif;font-size:46px;line-height:1;margin-bottom:3px;}}
  .ro40-cohort{{font-size:12px;color:var(--muted);margin-bottom:12px;}}
  .g-note{{background:rgba(79,142,247,.06);border:1px solid rgba(79,142,247,.2);border-radius:8px;padding:13px 18px;font-size:13px;color:var(--muted);line-height:1.7;margin-bottom:24px;}}
  .g-note strong{{color:var(--text);}}

  /* ── APPLICABILITY BAR ── */
  .appl-bar-wrap{{display:flex;align-items:center;gap:8px;margin-top:8px;}}
  .appl-bar-track{{flex:1;background:var(--surface2);border-radius:4px;height:5px;}}
  .appl-bar-fill{{height:5px;border-radius:4px;background:var(--accent);}}
  .appl-score-lbl{{font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);white-space:nowrap;}}

  /* ── ALL METHODS TABLE ── */
  .all-methods-tbl{{width:100%;border-collapse:collapse;font-size:12px;}}
  .all-methods-tbl th{{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:8px 10px;border-bottom:2px solid var(--border);text-align:left;}}
  .all-methods-tbl td{{padding:9px 10px;border-top:1px solid var(--border);vertical-align:middle;}}
  .all-methods-tbl tr:hover td{{background:rgba(79,142,247,.04);}}
  .all-methods-tbl .rank-col{{font-family:'DM Mono',monospace;font-weight:700;color:var(--muted);}}
  .all-methods-tbl .score-col{{font-family:'DM Mono',monospace;}}
  .all-methods-tbl .na-row td{{color:var(--muted);font-style:italic;}}
  .cat-badge{{font-size:9px;padding:2px 6px;border-radius:3px;font-family:'DM Mono',monospace;letter-spacing:1px;text-transform:uppercase;}}
  .cat-classic{{background:rgba(79,142,247,.12);color:#4f8ef7;}}
  .cat-growth{{background:rgba(0,200,150,.12);color:#00c896;}}
  .cat-meta{{background:rgba(255,215,0,.12);color:#ffd700;}}

  /* ── GLOSSARY ── */
  .glossary-section{{margin-bottom:36px;}}
  .glossary-cat{{font-family:'DM Mono',monospace;font-size:12px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border);}}
  .glossary{{display:grid;gap:0;}}
  .glossary dt{{font-weight:700;color:var(--text);font-size:14px;padding:14px 0 4px;border-top:1px solid var(--border);}}
  .glossary dt:first-of-type{{border-top:none;}}
  .glossary dd{{color:var(--muted);font-size:13px;line-height:1.7;padding-bottom:10px;margin-left:0;}}

</style>
</head>
<body>


<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('top8',this)">Top 8 Methods</button>
  <button class="tab-btn" onclick="switchTab('all',this)">All Methods</button>
  <button class="tab-btn" onclick="switchTab('growth',this)">Growth Deep-Dive</button>
  {backtest_tab_btn}
  <button class="tab-btn" onclick="switchTab('glossary',this)">Glossary</button>
</div>
<div id="tab-top8" class="tab-panel active">
<div class="hero">
  <div class="hero-label">Equity Valuation Report</div>
  <h1 class="hero-title">{company_name}<br><span style="color:var(--muted);font-size:0.5em;letter-spacing:2px;">{ticker}</span></h1>
  <p class="hero-subtitle">Multi-method analysis · {ts}</p>
  <div class="hero-meta">
    <div class="meta-item">
      <span class="meta-label">Current Price</span>
      <span class="meta-value">{price}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">DCF Fair Value</span>
      <span class="meta-value" style="color:var(--accent);">{dcf_fv}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Consensus Mean</span>
      <span class="meta-value" style="color:var(--accent);">{mean_fv}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Upside / Downside</span>
      <span class="meta-value {upside_cls}">{upside_str}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Overall Verdict</span>
      <span class="meta-value verdict">{verdict}</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- CONVICTION BANNER -->
  <div class="conviction-banner">
    <div class="conviction-score">{conv_count}/{top8_count}</div>
    <div class="conviction-info">
      <h2>{conviction} CONVICTION — {conv_count} of {top8_count} top methods converge</h2>
      <p>Methods are considered converging when their fair value estimate is within {thresh}% of the median estimate.
      {conv_interp}</p>
    </div>
    <div class="conviction-stats">
      <div class="cstat">
        <div class="cstat-label">Fair Value Range</div>
        <div class="cstat-value">{range_low} – {range_high}</div>
      </div>
      <div class="cstat">
        <div class="cstat-label">Spread</div>
        <div class="cstat-value">{spread_pct:.1f}%</div>
      </div>
    </div>
  </div>

  <!-- TOP 8 METHOD CARDS -->
  <section>
    <div class="section-label">Top 8 Most Applicable Methods</div>
    <div class="cards-grid">
      {method_cards}
    </div>
  </section>

  <!-- CONVERGENCE VISUALISER -->
  <section>
    <div class="section-label">Convergence Spectrum</div>
    <div class="bar-wrapper">
      {conv_bars}
    </div>
  </section>

  <!-- RANGE SUMMARY -->
  <section>
    <div class="section-label">Valuation Summary</div>
    <div class="range-summary">
      <div class="range-item">
        <span class="range-label">Bear Case</span>
        <span class="range-value red">{range_low}</span>
        <span class="range-sub">Lowest method estimate</span>
      </div>
      <div class="range-item">
        <span class="range-label">Consensus Mean</span>
        <span class="range-value yellow">{mean_fv}</span>
        <span class="range-sub">Average of top 8 methods</span>
      </div>
      <div class="range-item">
        <span class="range-label">Bull Case</span>
        <span class="range-value green">{range_high}</span>
        <span class="range-sub">Highest method estimate</span>
      </div>
      <div class="range-item">
        <span class="range-label">20% MoS Entry</span>
        <span class="range-value" style="color:var(--accent);">{mos_entry}</span>
        <span class="range-sub">Margin of safety on mean</span>
      </div>
      <div class="range-item">
        <span class="range-label">Market Cap</span>
        <span class="range-value" style="color:var(--text);font-size:22px;">{mktcap}</span>
        <span class="range-sub">At current price</span>
      </div>
      <div class="range-item">
        <span class="range-label">Est. FCF Growth</span>
        <span class="range-value" style="color:var(--text);font-size:22px;">{growth}</span>
        <span class="range-sub">Used across all methods</span>
      </div>
    </div>
  </section>

  <!-- DIVERGENCE ANALYSIS -->
  <section>
    <div class="section-label">Divergence Analysis</div>
    {divergence_html}
  </section>

  <!-- FUNDAMENTALS -->
  <section>
    <div class="section-label">Company Fundamentals</div>
    <div class="fundamentals">
      <div class="fund-table">
        <div class="fund-table-title">Income &amp; Cash Flow</div>
        <div class="fund-row"><span class="fund-key">Total Revenue (TTM)</span><span class="fund-val">{rev}</span></div>
        <div class="fund-row"><span class="fund-key">Net Income (TTM)</span><span class="fund-val">{ni}</span></div>
        <div class="fund-row"><span class="fund-key">Free Cash Flow (TTM)</span><span class="fund-val">{fcf}</span></div>
        <div class="fund-row"><span class="fund-key">Cash (est. from EV)</span><span class="fund-val">{cash_est}</span></div>
        <div class="fund-row"><span class="fund-key">EBITDA (derived)</span><span class="fund-val">{ebitda}</span></div>
        <div class="fund-row"><span class="fund-key">Operating Margin</span><span class="fund-val">{op_mar}</span></div>
        <div class="fund-row"><span class="fund-key">FCF Margin</span><span class="fund-val">{fcf_mar}</span></div>
        <div class="fund-row"><span class="fund-key">Gross Margin</span><span class="fund-val">{gross_mar}</span></div>
        <div class="fund-row"><span class="fund-key">EPS Diluted (TTM)</span><span class="fund-val">{eps}</span></div>
      </div>
      <div class="fund-table">
        <div class="fund-table-title">Valuation &amp; Balance Sheet</div>
        <div class="fund-row"><span class="fund-key">Market Cap</span><span class="fund-val">{mktcap}</span></div>
        <div class="fund-row"><span class="fund-key">P/E (TTM)</span><span class="fund-val">{pe}</span></div>
        <div class="fund-row"><span class="fund-key">P/FCF (TTM)</span><span class="fund-val">{pfcf}</span></div>
        <div class="fund-row"><span class="fund-key">EV/EBITDA (TV)</span><span class="fund-val">{ev_eb}</span></div>
        <div class="fund-row"><span class="fund-key">PEG Ratio</span><span class="fund-val">{peg}</span></div>
        <div class="fund-row"><span class="fund-key">Total Debt</span><span class="fund-val">{debt}</span></div>
        <div class="fund-row"><span class="fund-key">Debt / Equity</span><span class="fund-val">{d_e}</span></div>
        <div class="fund-row"><span class="fund-key">Beta (1yr)</span><span class="fund-val">{beta}</span></div>
      </div>
    </div>
  </section>

  <!-- BENCHMARKS -->
  <section>
    <div class="section-label">Live Industry Benchmarks — {sector_label} ({peer_count} peers)</div>
    <div class="range-summary">
      <div class="range-item">
        <span class="range-label">{sector_label} P/E</span>
        <span class="range-value" style="color:var(--text);font-size:24px;">{bm_pe}x</span>
      </div>
      <div class="range-item">
        <span class="range-label">{sector_label} P/FCF</span>
        <span class="range-value" style="color:var(--text);font-size:24px;">{bm_pfcf}x</span>
      </div>
      <div class="range-item">
        <span class="range-label">{sector_label} EV/EBITDA</span>
        <span class="range-value" style="color:var(--text);font-size:24px;">{bm_evebitda}x</span>
      </div>
    </div>
  </section>

</div>

</div><!-- /tab-top8 -->

<div id="tab-all" class="tab-panel">
<div class="container" style="max-width:1100px;margin:0 auto;padding:40px 24px;">
  <div class="section-label">All Valuation Methods — Ranked by Applicability</div>
  <p style="color:var(--muted);font-size:13px;margin-bottom:24px;">
    Models ranked by applicability score (0–100) combining academic precision, data quality, and company fit.
    Top 8 are displayed in the "Top 8" tab. All models with a fair value are shown here.
  </p>
  {all_methods_table}
</div>
</div><!-- /tab-all -->

<div id="tab-growth" class="tab-panel">
  <div class="container" style="padding-top:48px;">
    {growth_section_html}
  </div>
</div><!-- /tab-growth -->

{backtest_tab_panel}

<div id="tab-glossary" class="tab-panel">
  <div class="container" style="padding-top:48px;">
    <div class="section-label">Glossary — Terms &amp; Acronyms</div>

    <div class="glossary-section">
      <h3 class="glossary-cat">Valuation Methods</h3>
      <dl class="glossary">
        <dt>DCF (Discounted Cash Flow)</dt>
        <dd>Projects a company's future free cash flows over a set period, discounts them back to present value using WACC, then adds a terminal value for all cash flows beyond the projection period. The sum, minus net debt, divided by shares outstanding gives an intrinsic value per share.</dd>
        <dt>P/E (Price-to-Earnings)</dt>
        <dd>Compares the stock price to earnings per share. This report applies target, sector median, and conservative P/E multiples to TTM EPS and averages the three to estimate fair value.</dd>
        <dt>P/FCF (Price-to-Free-Cash-Flow)</dt>
        <dd>Similar to P/E but uses free cash flow per share instead of earnings. FCF is often considered a purer measure of profitability because it accounts for capital expenditures and is harder to manipulate with accounting choices.</dd>
        <dt>EV/EBITDA (Enterprise Value to EBITDA)</dt>
        <dd>Values the entire enterprise (equity + net debt) relative to EBITDA. Useful for comparing companies with different capital structures because it looks through leverage. The implied equity value per share is (EBITDA × multiple − net debt) / shares.</dd>
        <dt>Reverse DCF</dt>
        <dd>Works backwards from the current stock price to determine what FCF growth rate the market is implicitly pricing in. Useful for asking: "Is the market's expectation reasonable?"</dd>
        <dt>Forward PEG</dt>
        <dd>Applies the Peter Lynch PEG ratio concept using next-twelve-month EPS. Fair value = NTM EPS × growth% × PEG multiple. A PEG of 1.0 means the P/E equals the growth rate (considered "fair"); quality moats can justify 1.5–2.0×.</dd>
        <dt>EV/NTM Revenue</dt>
        <dd>Values the enterprise on forward revenue using multiples tiered by gross margin and growth rate, blended with the stock's current market EV/Revenue. Best suited for high-growth companies whose earnings don't yet reflect their earning power.</dd>
        <dt>TAM Scenario (Total Addressable Market)</dt>
        <dd>Estimates year-5 revenue based on a percentage capture of the total addressable market, applies a net margin to get earnings, values those at a terminal P/E, and discounts back to today. Bear/base/bull scenarios reflect different market share assumptions.</dd>
        <dt>Rule of 40</dt>
        <dd>A SaaS/software quality benchmark: revenue growth% + FCF margin% should exceed 40 for a healthy company. The score determines which peer-group EV/Revenue cohort applies, which then drives a fair value estimate.</dd>
      </dl>
    </div>

    <div class="glossary-section">
      <h3 class="glossary-cat">Financial Metrics &amp; Inputs</h3>
      <dl class="glossary">
        <dt>EPS (Earnings Per Share)</dt>
        <dd>Net income divided by diluted shares outstanding. TTM = trailing twelve months. NTM / Forward EPS = analyst consensus estimate for the next twelve months or fiscal year.</dd>
        <dt>FCF (Free Cash Flow)</dt>
        <dd>Cash generated by operations minus capital expenditures. Represents the cash actually available to shareholders, debt holders, and for reinvestment.</dd>
        <dt>EBITDA</dt>
        <dd>Earnings Before Interest, Taxes, Depreciation, and Amortisation. A proxy for operating cash generation before capital structure and non-cash charges. When not directly available, this report estimates it from EV/EBITDA, operating income, or net income.</dd>
        <dt>EV (Enterprise Value)</dt>
        <dd>Market capitalisation + total debt − cash. Represents the theoretical takeover price of the entire business. Used as the numerator in EV-based multiples.</dd>
        <dt>Net Debt</dt>
        <dd>Total debt minus cash and equivalents. A positive net debt means the company owes more than it holds in cash; a negative net debt means it has more cash than debt.</dd>
        <dt>Market Cap (Market Capitalisation)</dt>
        <dd>Share price × total shares outstanding. The market's current valuation of a company's equity.</dd>
        <dt>Gross Margin</dt>
        <dd>Revenue minus cost of goods sold, divided by revenue (expressed as %). Indicates how much profit a company retains from each dollar of revenue before operating expenses.</dd>
        <dt>Operating Margin</dt>
        <dd>Operating income divided by revenue (expressed as %). Measures profitability from core operations after all operating costs but before interest and taxes.</dd>
        <dt>FCF Margin</dt>
        <dd>Free cash flow divided by revenue (expressed as %). Shows what percentage of revenue converts into free cash.</dd>
        <dt>Net Margin</dt>
        <dd>Net income divided by revenue (expressed as %). The bottom-line profitability after all expenses, interest, and taxes.</dd>
        <dt>D/E (Debt-to-Equity)</dt>
        <dd>Total debt divided by total shareholders' equity. Measures financial leverage — higher values indicate more debt-financed operations.</dd>
        <dt>Beta</dt>
        <dd>Measures a stock's volatility relative to the overall market. Beta of 1.0 = moves with the market; &gt;1.0 = more volatile; &lt;1.0 = less volatile. Used in WACC calculation.</dd>
        <dt>PEG Ratio (Price/Earnings-to-Growth)</dt>
        <dd>P/E ratio divided by the expected earnings growth rate. A PEG of 1.0 is considered fairly valued; below 1.0 may be undervalued relative to growth; above 2.0 may be overvalued.</dd>
        <dt>NTM (Next Twelve Months)</dt>
        <dd>A forward-looking time horizon. NTM EPS and NTM Revenue are analyst consensus estimates for the next twelve months.</dd>
        <dt>TTM (Trailing Twelve Months)</dt>
        <dd>The most recent twelve-month period of actual reported financial data. Used for backward-looking metrics like TTM P/E.</dd>
        <dt>D&amp;A (Depreciation &amp; Amortisation)</dt>
        <dd>Non-cash charges that reduce reported earnings but don't consume cash. Added back to operating income to approximate EBITDA.</dd>
      </dl>
    </div>

    <div class="glossary-section">
      <h3 class="glossary-cat">Valuation Concepts</h3>
      <dl class="glossary">
        <dt>WACC (Weighted Average Cost of Capital)</dt>
        <dd>The blended rate of return a company must earn to satisfy all capital providers. Used as the discount rate in DCF. This report calculates it using the full textbook formula: WACC = (E/V)×Ke + (D/V)×Kd×(1−t), where Ke is the CAPM cost of equity (Rf + β×ERP), Kd is the effective cost of debt (interest expense / total debt), and t is the effective tax rate. Weights are based on market-cap equity and total debt. Falls back gracefully if any component is unavailable. Can be overridden with the --wacc flag.</dd>
        <dt>CAPM (Capital Asset Pricing Model)</dt>
        <dd>A model that estimates the expected return of an equity investment: Expected Return = Risk-Free Rate + Beta × Equity Risk Premium. Used here to derive WACC.</dd>
        <dt>Risk-Free Rate (Rf)</dt>
        <dd>The theoretical return on a zero-risk investment, typically proxied by the yield on 10-year US Treasury bonds. Currently set at {rf_pct}% in this report.</dd>
        <dt>ERP (Equity Risk Premium)</dt>
        <dd>The additional return investors demand for holding equities over risk-free bonds. Currently set at {erp_pct}% in this report.</dd>
        <dt>Terminal Value</dt>
        <dd>The value of all cash flows beyond the explicit projection period, assuming a perpetual growth rate (currently {tgr_pct}%). Often represents 60–85% of total DCF value — when it exceeds 85%, the model is heavily dependent on long-term assumptions.</dd>
        <dt>MoS (Margin of Safety)</dt>
        <dd>A discount applied to the calculated fair value to create a buffer against estimation errors. This report uses a {mos_pct}% margin of safety. If fair value is $100, the MoS price is ${mos_ex}.</dd>
        <dt>Intrinsic Value</dt>
        <dd>The estimated "true" worth of a stock based on fundamental analysis, independent of its current market price.</dd>
        <dt>Fair Value</dt>
        <dd>The estimated price at which a stock is neither over- nor under-valued. Each valuation method produces its own fair value estimate.</dd>
        <dt>Convergence</dt>
        <dd>When multiple independent valuation methods produce similar fair value estimates (within {conv_thresh}% of each other). Higher convergence = higher confidence in the estimate.</dd>
        <dt>Conviction Level</dt>
        <dd>A qualitative signal based on how many methods converge: HIGH (all 4), MODERATE (3 of 4), or LOW (2 or fewer). Higher conviction means the estimate is more robust to changes in any single method's assumptions.</dd>
      </dl>
    </div>

    <div class="glossary-section">
      <h3 class="glossary-cat">Report-Specific Labels</h3>
      <dl class="glossary">
        <dt>CONVERGING</dt>
        <dd>Badge shown on methods whose fair value estimate falls within {conv_thresh}% of the consensus median. These methods agree with each other.</dd>
        <dt>OUTLIER</dt>
        <dd>Badge shown on methods whose fair value diverges more than {conv_thresh}% from the consensus median. The divergence analysis section explains why.</dd>
        <dt>LOW CONFIDENCE</dt>
        <dd>Badge shown on methods where the reliability assessment has detected issues that make the estimate less trustworthy for this specific stock. Flagged cards are dimmed — hover to reveal full detail.</dd>
        <dt>Industry / Sector Median</dt>
        <dd>The median multiple (P/E, P/FCF, or EV/EBITDA) calculated from the tightest available peer set. The script tries three tiers in order: (1) industry peers (e.g. "Semiconductors" for NVDA — the most specific and most meaningful), (2) sector peers (e.g. "Technology") if the industry cohort is too small, and (3) broad-market top-200 as a last resort. The label shown in the report tells you which tier was used.</dd>
        <dt>Growth-Adjusted Multiple</dt>
        <dd>A target or conservative multiple scaled to the company's estimated growth rate using PEG-style logic, so faster-growing companies aren't penalised by static average multiples.</dd>
      </dl>
    </div>

    <div class="glossary-section">
      <h3 class="glossary-cat">Backtesting — Concepts &amp; Metrics</h3>
      <dl class="glossary">
        <dt>Backtest</dt>
        <dd>A historical simulation that tests how well a valuation model would have predicted a stock's price. This report fetches real historical fundamentals from SEC EDGAR filings and real historical prices from Stooq, then re-runs every valuation method using only the data that was publicly available on the backtest start date — no lookahead bias.</dd>
        <dt>Backtest Start Date</dt>
        <dd>The date N trading days ago that defines the beginning of the test window. The most recent annual 10-K filing publicly available on that date (applying a 75-day filing lag) is used as the fundamental input for all methods.</dd>
        <dt>10-K Filing Lag (75 days)</dt>
        <dd>Annual reports (10-K) are filed with the SEC 60–90 days after a company's fiscal year ends. This report applies a 75-day lag so it only uses data that was genuinely available to investors on each backtest date — preventing accidental lookahead bias.</dd>
        <dt>XBRL (eXtensible Business Reporting Language)</dt>
        <dd>A structured data format used by all SEC filers since 2009. The SEC EDGAR XBRL API (data.sec.gov) exposes these filings as machine-readable JSON, which this report uses to retrieve historical revenue, FCF, net income, margins, and debt.</dd>
        <dt>MAPE (Mean Absolute Percentage Error)</dt>
        <dd>The average of |fair_value − actual_price| / actual_price × 100, computed across every trading day in the backtest window. Measures how far the method's estimate strayed from reality on a typical day. Lower = better. A MAPE of 10% means the method was off by 10% on average.</dd>
        <dt>Accuracy Score</dt>
        <dd>Defined as 100 − MAPE. A score of 90 means the method's fair value was within 10% of the actual price on average. Scores above 85 are rated EXCELLENT; 70–85 GOOD; 50–70 MODERATE; below 50 POOR.</dd>
        <dt>Directional Signal (BUY / SELL / HOLD)</dt>
        <dd>The signal implied by the method's fair value at the backtest start date relative to the price on that date. BUY = fair value &gt; start price by more than 5%; SELL = fair value &lt; start price by more than 5%; HOLD = within 5%. This is what a fundamental investor using only that method would have done on day 1.</dd>
        <dt>Direction ✓ (Directional Accuracy)</dt>
        <dd>Whether the method's directional signal turned out to be correct. BUY is correct if the stock rose more than 2% over the window; SELL is correct if it fell more than 2%; HOLD is correct if it moved less than 10% in either direction.</dd>
        <dt>Evolving Fair Value Curve</dt>
        <dd>Each method's fair value is re-computed for every trading day using whichever annual filing was available on that date. The result is a stepped curve that changes when a new 10-K becomes available, rather than a flat horizontal line. This is what makes the backtest meaningful — the model's estimate evolves as the company's fundamentals change.</dd>
        <dt>Filing Boundary (vertical dashed lines on chart)</dt>
        <dd>Vertical markers on the backtest chart showing the dates when a new annual 10-K filing became available (i.e., the fiscal year-end date plus 75 days). Fair value lines step to new levels at these boundaries as the updated fundamentals are incorporated.</dd>
        <dt>Stooq</dt>
        <dd>A free financial data provider (stooq.com) used by this report to retrieve historical daily prices. No API key is required. US equities use the symbol format <em>aapl.us</em>.</dd>
        <dt>SEC EDGAR</dt>
        <dd>The U.S. Securities and Exchange Commission's Electronic Data Gathering, Analysis, and Retrieval system. All public companies file their financial reports here. This report uses the free XBRL API at data.sec.gov to retrieve historical fundamental data with no API key.</dd>
      </dl>
    </div>

  </div>
</div><!-- /tab-glossary -->

<script>
function switchTab(n,b){{
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active');
  b.classList.add('active');
}}

// Backtest chart line toggle
// gid = the sanitised method id, e.g. "btline-dcf"
// State is tracked on the card element via data-visible attribute.
function btNoLine(methodName) {{
  var msg = document.getElementById('bt-noline-msg');
  if (!msg) {{
    msg = document.createElement('div');
    msg.id = 'bt-noline-msg';
    msg.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1e2030;border:1px solid #333;border-radius:8px;padding:10px 20px;font-size:12px;color:#aaa;z-index:9999;pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(msg);
  }}
  msg.textContent = methodName + ': no historical chart line (uses current-period data only)';
  msg.style.opacity = '1';
  clearTimeout(msg._timer);
  msg._timer = setTimeout(function(){{ msg.style.opacity = '0'; }}, 2500);
}}

function btToggleLine(gid) {{
  var line = document.getElementById(gid);
  var card = document.getElementById('card-' + gid);
  var eye  = document.getElementById('eye-'  + gid);
  if (!line || !card) return;

  var isVisible = card.getAttribute('data-visible') !== 'false';

  if (isVisible) {{
    // Hide: dim the card, strike the eye, hide the SVG group
    line.style.display = 'none';
    card.style.opacity = '0.35';
    card.style.boxShadow = 'none';
    if (eye) eye.innerHTML = '🙈';
    card.setAttribute('data-visible', 'false');
  }} else {{
    // Show: restore card, restore SVG group
    line.style.display = '';
    card.style.opacity = '1';
    card.style.boxShadow = '0 0 0 2px rgba(255,255,255,0.15)';
    if (eye) eye.innerHTML = '👁';
    card.setAttribute('data-visible', 'true');
  }}
}}
</script>

<div class="footer">
  Generated {ts} · Data source: TradingView Screener · For educational purposes only.<br>
  This report is not investment advice. Always verify data against SEC filings (10-K / 10-Q) before making investment decisions.
</div>

</body>
</html>
""".format(
        ticker=ticker,
        name=name,
        company_name=company_name,
        ts=ts,
        price=mo(price),
        dcf_fv=mo(top_8[0]["fair_value"] if top_8 else (results[0]["fair_value"] if results else None)),
        mean_fv=mo(conv["mean"]),
        upside_str=(("+" if conv["upside"] >= 0 else "") + format(conv["upside"], ".1f") + "%"),
        upside_cls=("up" if conv["upside"] >= 0 else "down"),
        verdict=conv["verdict"],
        verdict_color=verdict_color,
        conviction=conv["conviction"],
        conviction_color=conv["conviction_color"],
        conv_count=len(conv["converging"]),
        thresh=int(CONVERGENCE_THRESHOLD * 100),
        conv_interp=(
            "Strong signal — when all top methods agree, the estimate is robust to assumption changes." if len(conv["converging"]) >= len(top_8)
            else "Moderate signal — review the outlier method's assumptions carefully." if len(conv["converging"]) >= len(top_8) // 2
            else "Weak signal — significant divergence exists. Treat any single estimate with caution."
        ),
        top8_count=len(top_8),
        all_methods_table=_build_all_methods_table(top_8, remainder, applicability_scores, method_colors, d["price"], bt=bt),
        range_low=mo(conv["low"]),
        range_high=mo(conv["high"]),
        spread_pct=conv["spread_pct"],
        method_cards=method_cards_html,
        conv_bars=conv_bars,
        mos_entry=mo(conv["mean"] * (1 - MARGIN_OF_SAFETY)),
        mktcap=B(d["market_cap"]),
        growth=pc(d["est_growth"]),
        divergence_html=div_html,
        rev=B(d["revenue"]),
        ni=B(d["net_income"]),
        fcf=B(d["fcf"]),
        cash_est=B(d["cash"]) + "  (" + d.get("cash_source","est") + ")",
        ebitda=B(d["ebitda"]),
        op_mar=(format(d["op_margin"], ".1f") + "%" if d["op_margin"] else "N/A"),
        fcf_mar=pc(d["fcf_margin"]),
        gross_mar=(format(d["gross_margin"], ".1f") + "%" if d["gross_margin"] else "N/A"),
        eps=mo(d["eps"]),
        pe=xf(d["current_pe"]),
        pfcf=xf(d["current_pfcf"]),
        ev_eb=xf(d["tv_ev_ebitda"]),
        peg=(format(d["peg"], ".2f") + "x" if d["peg"] else "N/A"),
        debt=M(d["total_debt"]),
        d_e=xf(d["debt_equity"]),
        beta=format(d["beta"], ".3f"),
        bm_pe=format(benchmarks["pe"], ".1f"),
        bm_pfcf=format(benchmarks["pfcf"], ".1f"),
        bm_evebitda=format(benchmarks["ev_ebitda"], ".1f"),
        sector_label=benchmarks.get("sector_name", "Broad Market"),
        peer_count=benchmarks.get("peer_count", 0),
        growth_section_html=_build_growth_html(d, gr, reliability),
        rf_pct=format(RISK_FREE_RATE * 100, ".1f"),
        erp_pct=format(EQUITY_RISK_PREMIUM * 100, ".1f"),
        tgr_pct=format(TERMINAL_GROWTH_RATE * 100, ".1f"),
        mos_pct=int(MARGIN_OF_SAFETY * 100),
        mos_ex=format(100 * (1 - MARGIN_OF_SAFETY), ".0f"),
        conv_thresh=int(CONVERGENCE_THRESHOLD * 100),
        backtest_tab_btn=(
            '<button class="tab-btn" onclick="switchTab(\'backtest\',this)">Backtesting</button>'
            if bt and "error" not in bt else
            '<button class="tab-btn" onclick="switchTab(\'backtest\',this)">Backtesting</button>'
        ),
        backtest_tab_panel=(
            '<div id="tab-backtest" class="tab-panel">'
            '<div class="container" style="padding-top:48px;">'
            + _build_backtest_html(bt, d, top_8=top_8, applicability_scores=applicability_scores) +
            '</div></div><!-- /tab-backtest -->'
        ) if bt is not None else (
            '<div id="tab-backtest" class="tab-panel">'
            '<div class="container" style="padding-top:48px;">'
            '<div class="section-label">Backtesting</div>'
            '<p style="color:var(--muted);font-size:14px;">Run with <code>--backtest DAYS</code> to enable this tab.</p>'
            '</div></div>'
        ),
    )

    return html


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-method equity valuation report",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "ticker", nargs="?", default="AAPL",
        help="Ticker symbol (default: AAPL)"
    )
    parser.add_argument(
        "--wacc", type=float, default=None, metavar="RATE",
        help=(
            "Override the auto-calculated WACC for DCF methods.\n"
            "Provide as a decimal: --wacc 0.09 means 9%%.\n"
            "Auto WACC = Rf + beta x ERP (often 10-11%% for large-caps).\n"
            "Analyst consensus for blue-chips is typically 8-10%%.\n"
            "Example: python valuationMaster.py AAPL --wacc 0.09"
        )
    )
    parser.add_argument(
        "--backtest", type=int, default=1000, metavar="DAYS",
        help=(
            "Number of trading days to include in the historical backtest (default: 1000).\n"
            "Set to 0 to skip the backtest entirely.\n"
            "Example: python valuationMaster.py AAPL --backtest 252  (≈ 1 year)\n"
            "Example: python valuationMaster.py AAPL --backtest 0    (skip backtest)"
        )
    )
    args   = parser.parse_args()
    ticker = args.ticker.upper()

    print("\n" + "=" * 56)
    print("  MASTER VALUATION REPORT — " + ticker)
    print("=" * 56)
    if args.wacc:
        print(f"  WACC override: {args.wacc:.2%}  (auto-calc disabled)")

    print("\n[1/9] Fetching fundamental data from TradingView...")
    d = fetch_tv_data(ticker)
    if args.wacc:
        d["wacc_override"] = args.wacc
        auto_wacc, auto_source = calculate_wacc(d)
        print(f"  Full WACC would have been: {auto_wacc:.2%} ({auto_source})  ->  overridden to {args.wacc:.2%}")
    else:
        auto_wacc, auto_source = calculate_wacc(d)
        print(f"  WACC: {auto_wacc:.2%}  ({auto_source})")
    print("  OK: Price=" + mo(d["price"]) + " | FCF=" + B(d["fcf"]) + " | EPS=" + mo(d["eps"]))

    print("[2/9] Fetching extended yfinance data (book value, ROIC, dividends, history)...")
    d["ext"] = fetch_yfinance_extended(ticker)

    print("[3/9] Fetching live industry/sector benchmarks & ERG peer calibration data...")
    sector   = d.get("sector")
    industry = d.get("industry")
    if industry:
        print("  Industry detected: {} ({}) — fetching industry peer medians...".format(industry, sector or "?"))
    elif sector:
        print("  Sector detected: {} — fetching sector peer medians...".format(sector))
    benchmarks = fetch_market_benchmarks(sector, industry)
    print("  Benchmark source: " + benchmarks.get("sector_name", "Broad Market")
          + " (" + str(benchmarks.get("peer_count", 0)) + " peers)")
    print("  P/E median=" + str(benchmarks["pe"]) + "x | P/FCF median=" + str(benchmarks["pfcf"]) + "x | EV/EBITDA median=" + str(benchmarks["ev_ebitda"]) + "x")
    erg_peer_data = fetch_peer_erg_data(sector, industry)
    if erg_peer_data.get("peer_count", 0) > 0:
        print("  ERG peers:  {} companies  |  p25={:.3f}  median={:.3f}  p75={:.3f}  ({})".format(
            erg_peer_data["peer_count"],
            erg_peer_data["p25"], erg_peer_data["median"], erg_peer_data["p75"],
            erg_peer_data["sector_name"],
        ))
    else:
        print("  ERG peers:  No data — ERG will use Earnings Build only")

    print("[4/9] Running classic valuation models...")
    results = []
    gr = {}

    if d["fcf"] and d["fcf"] > 0 and d["shares"]:
        r = run_dcf(d)
        results.append(r)
        print("  DCF:              " + mo(r["fair_value"]))
    else:
        print("  DCF:              SKIPPED (no positive FCF)")

    if d["fcf_per_share"] and d["fcf_per_share"] > 0:
        r = run_pfcf(d, benchmarks)
        results.append(r)
        print("  P/FCF:            " + mo(r["fair_value"]))
    else:
        print("  P/FCF:            SKIPPED (no positive FCF per share)")

    if d["eps"] and d["eps"] > 0:
        r = run_pe(d, benchmarks)
        results.append(r)
        print("  P/E:              " + mo(r["fair_value"]))
    else:
        print("  P/E:              SKIPPED (no positive EPS)")

    if d["ebitda"] and d["ebitda"] > 0 and d["shares"]:
        r = run_ev_ebitda(d, benchmarks)
        results.append(r)
        print("  EV/EBITDA:        " + mo(r["fair_value"]))
    else:
        print("  EV/EBITDA:        SKIPPED (no positive EBITDA)")

    if d["fcf"] and d["fcf"] > 0 and d["shares"]:
        r = run_three_stage_dcf(d)
        if r:
            results.append(r)
            print("  Three-Stage DCF:  " + mo(r["fair_value"]))
    else:
        print("  Three-Stage DCF:  SKIPPED (no positive FCF)")

    try:
        r = run_monte_carlo_dcf(d)
        if r:
            results.append(r)
            print("  Monte Carlo DCF:  " + mo(r["fair_value"]) +
                  "  [p10=" + mo(r.get("p10")) + " p90=" + mo(r.get("p90")) + "]")
        else:
            print("  Monte Carlo DCF:  SKIPPED")
    except Exception:
        print("  Monte Carlo DCF:  SKIPPED (numpy unavailable)")

    if d.get("fcf_per_share") and d["fcf_per_share"] > 0:
        r = run_fcf_yield(d)
        if r:
            results.append(r)
            print("  FCF Yield:        " + mo(r["fair_value"]))
        else:
            print("  FCF Yield:        SKIPPED")
    else:
        print("  FCF Yield:        SKIPPED (no positive FCF per share)")

    ext = d.get("ext", {})
    if ext.get("book_value_ps") and d.get("eps") and d["eps"] > 0:
        r = run_rim(d)
        if r:
            results.append(r)
            print("  RIM:              " + mo(r["fair_value"]))
        else:
            print("  RIM:              SKIPPED")
    else:
        print("  RIM:              SKIPPED (no book value or EPS)")

    if ext.get("roic") is not None and ext.get("stockholders_equity"):
        r = run_roic_excess_return(d)
        if r:
            results.append(r)
            print("  ROIC Excess Ret:  " + mo(r["fair_value"]))
        else:
            print("  ROIC Excess Ret:  SKIPPED")
    else:
        print("  ROIC Excess Ret:  SKIPPED (no ROIC data)")

    _ncav_skip_reason = "Insufficient data"
    if ext.get("total_current_assets") is not None and ext.get("total_liabilities") is not None and d.get("shares"):
        r = run_ncav(d)
        if r:
            results.append(r)
            gr["ncav"] = r   # store in gr for HTML access
            print("  NCAV:             " + mo(r["fair_value"]) + "  (" + r.get("graham_verdict","") + ")")
        else:
            # Data available but NCAV is negative — compute descriptive reason
            ncav_ps_raw = (ext["total_current_assets"] - ext["total_liabilities"]) / d["shares"]
            _ncav_skip_reason = (
                "Negative NCAV (${:.2f}/sh): stock trades above liquidation floor"
                .format(round(ncav_ps_raw, 2))
            )
            print("  NCAV:             Not applicable — " + _ncav_skip_reason)
    else:
        print("  NCAV:             SKIPPED (no current assets / liabilities)")

    if all(ext.get(k) for k in ["hist_pe_5y"]) or ext.get("hist_pfcf_5y") or ext.get("hist_eveb_5y"):
        r = run_mean_reversion(d)
        if r:
            results.append(r)
            gr["mean_reversion"] = r   # also store in gr so Growth Deep-Dive tab can display it
            print("  Mean Reversion:   " + mo(r["fair_value"]) +
                  "  (" + str(r.get("n_components", 0)) + " components)")
        else:
            print("  Mean Reversion:   SKIPPED")
    else:
        print("  Mean Reversion:   SKIPPED (no historical multiples)")

    if len(results) < 2:
        print("\nERROR: Need at least 2 valid methods to run convergence analysis.")
        sys.exit(1)

    print("[5/9] Running growth analysis methods...")
    r = run_reverse_dcf(d)
    if r: gr["reverse_dcf"] = r;    print("  Reverse DCF:    implied " + format(r["implied_growth"]*100,".1f") + "% growth")
    else: print("  Reverse DCF:    SKIPPED (no positive FCF)")
    r = run_forward_peg(d)
    if r: gr["forward_peg"] = r;    print("  Fwd PEG:        $" + format(r["fair_value"],",.2f"))
    else: print("  Fwd PEG:        SKIPPED (no EPS)")
    r = run_ev_ntm_revenue(d)
    if r: gr["ev_ntm_revenue"] = r; print("  EV/NTM Rev:     $" + format(r["fair_value"],",.2f"))
    else: print("  EV/NTM Rev:     SKIPPED (no revenue)")
    r = run_tam_scenario(d)
    if r: gr["tam_scenario"] = r;   print("  TAM Scenario:   $" + format(r["fair_value"],",.2f") + " (base)")
    else: print("  TAM Scenario:   SKIPPED (no revenue)")
    r = run_rule_of_40(d)
    if r: gr["rule_of_40"] = r;     print("  Rule of 40:     " + str(r["ro40"]) + "  " + r["cohort"])
    else: print("  Rule of 40:     SKIPPED (no revenue)")
    r = run_erg_valuation(d, erg_peer_data)
    if r: gr["erg"] = r;            print("  ERG:            $" + format(r["fair_value"],",.2f") + " (blended base)")
    else: print("  ERG:            SKIPPED (no revenue)")

    # New growth models
    r = run_scurve_tam(d)
    if r: gr["scurve_tam"] = r;     print("  S-Curve TAM:    $" + format(r["fair_value"],",.2f") + " (base)")
    else: print("  S-Curve TAM:    SKIPPED")
    r = run_pie(d)
    if r: gr["pie"] = r;            print("  PIE:            $" + format(r["fair_value"],",.2f"))
    else: print("  PIE:            SKIPPED")
    if ext.get("dividends_per_share") and ext["dividends_per_share"] > 0:
        r = run_ddm_hmodel(d)
        if r: gr["ddm"] = r;        print("  DDM H-Model:    $" + format(r["fair_value"],",.2f"))
        else: print("  DDM H-Model:    SKIPPED")
    else:
        print("  DDM H-Model:    SKIPPED (no dividends)")

    print("[6/9] Running Multi-Factor price target...")
    r = run_multifactor_price_target(d, benchmarks)
    if r: gr["multifactor"] = r;    print("  Multi-Factor:   $" + format(r["fair_value"],",.2f"))
    else: print("  Multi-Factor:   SKIPPED")

    print("[7/9] Assessing method reliability...")
    reliability = assess_reliability(d, results, gr)
    flagged = [m for m, f in reliability.items() if not f["reliable"]]
    if flagged:
        print("  ⚠ Low confidence: " + ", ".join(flagged))
    else:
        print("  All methods appear reliable for this stock.")

    print("[8/9] Scoring applicability & ranking all models...")
    # Build unified list of all models that produced a fair value
    # Deduplicate by method name (NCAV and Mean Reversion are stored in both results and gr)
    all_model_results = list(results)
    _existing_methods = {r2["method"] for r2 in all_model_results}
    for gkey, gval in gr.items():
        if gval and isinstance(gval, dict) and gval.get("fair_value") is not None:
            if gval.get("method") not in _existing_methods:
                all_model_results.append(gval)
                _existing_methods.add(gval["method"])

    reliability_flags = {m: f.get("flags", []) for m, f in reliability.items()}
    applicability_scores = {}
    for r2 in all_model_results:
        mname = r2.get("method", "")
        applicability_scores[mname] = score_model_applicability(mname, r2, d, reliability_flags.get(mname, []))

    sorted_models = sorted(all_model_results, key=lambda r2: applicability_scores.get(r2.get("method",""), 0), reverse=True)
    top_8   = sorted_models[:8]
    remainder = sorted_models[8:]

    # Also collect models that returned None (for All Methods table)
    # Note: Bayesian Ensemble is excluded here — it runs in step [9/9] and is handled there
    _skip_reasons = {"NCAV": _ncav_skip_reason}
    skipped_methods = []
    for mname in ["Three-Stage DCF","Monte Carlo DCF","FCF Yield","RIM","ROIC Excess Return",
                   "NCAV","Mean Reversion","S-Curve TAM","PIE","DDM","Multi-Factor"]:
        if not any(r2.get("method") == mname for r2 in all_model_results):
            reason = _skip_reasons.get(mname, "Insufficient data")
            skipped_methods.append({"method": mname, "fair_value": None,
                                     "mos_value": None, "skip_reason": reason})

    print("  Top 8 models: " + ", ".join(r2["method"] for r2 in top_8))

    print("[9/9] Running Bayesian ensemble (meta-model — requires all others)...")
    earnings_surp = ext.get("earnings_surprise_pct")
    r = run_bayesian_ensemble(d, all_model_results, applicability_scores, earnings_surp)
    if r:
        gr["bayesian"] = r
        bayes_score = score_model_applicability("Bayesian Ensemble", r, d, reliability_flags.get("Bayesian Ensemble", []))
        applicability_scores["Bayesian Ensemble"] = bayes_score
        all_model_results.append(r)
        sorted_models = sorted(all_model_results, key=lambda r2: applicability_scores.get(r2.get("method",""), 0), reverse=True)
        top_8   = sorted_models[:8]
        remainder = sorted_models[8:]
        print("  Bayesian Ensemble: $" + format(r["fair_value"],",.2f") + "  (n_models=" + str(r.get("n_models_used",0)) + ")")
    else:
        print("  Bayesian Ensemble: SKIPPED (fewer than 3 valid models)")
        skipped_methods.append({"method": "Bayesian Ensemble", "fair_value": None,
                                 "mos_value": None, "skip_reason": "Insufficient data"})

    conv = analyse_convergence([r2 for r2 in top_8 if r2.get("fair_value")], d["price"])
    print("  Conviction: " + conv["conviction"] + " (" + str(len(conv["converging"])) + "/" + str(len(top_8)) + " top methods converge)")
    print("  Consensus mean fair value: " + mo(conv["mean"]))
    print("  Verdict: " + conv["verdict"])

    bt = None
    if args.backtest > 0:
        backtest_days = min(args.backtest, 365 * 5)  # cap at 5 years (~1260 trading days)
        print("\n[Backtest] Running {}-day price backtest...".format(backtest_days))
        bt = run_backtest(d, results, gr, backtest_days, erg_peer_data)
        if "error" in bt:
            print("  ⚠ Backtest warning: " + bt["error"])
        else:
            print("  Best method:  " + bt["best_method"]
                  + " (accuracy: " + format(bt["method_stats"][bt["best_method"]]["accuracy_score"], ".1f") + "/100)")
            print("  Worst method: " + bt["worst_method"])
            print("  {}-day return: {}%".format(backtest_days, bt["price_change_pct"]))

    print("\nGenerating HTML report...")
    html = generate_html(d, results, conv, benchmarks, gr, reliability, bt,
                         applicability_scores=applicability_scores,
                         top_8=top_8, remainder=remainder + skipped_methods)

    outdir = "valuationData"
    os.makedirs(outdir, exist_ok=True)
    date_prefix = datetime.datetime.now().strftime("%Y_%m_%d")
    outfile = os.path.join(outdir, date_prefix + "_valuation_" + ticker + ".html")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    print("  Saved: " + outfile)
    print("\nOpening report in browser...")
    webbrowser.open("file://" + os.path.abspath(outfile))
    print("\nDone.")


if __name__ == "__main__":
    main()
