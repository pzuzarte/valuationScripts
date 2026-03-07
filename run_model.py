"""
run_model.py — Quick single-model valuation at the command line
===============================================================
Fetches data from yfinance, builds the canonical d-dict, runs any
model from valuation_models.py, and prints the result to the terminal.

Usage
-----
  python run_model.py TICKER MODEL [--wacc RATE] [--growth RATE]

  TICKER   Yahoo Finance ticker symbol (e.g. NVDA, AAPL, MSFT)
  MODEL    Model name or alias (see list below). Use 'all' to run everything.

Models & aliases
----------------
  dcf                  Classic 2-stage DCF
  three-stage / 3dcf   Three-Stage DCF (12-year)
  monte-carlo / mc     Monte Carlo DCF (5000 sims)
  pfcf                 Price / Free Cash Flow
  pe                   Price / Earnings
  ev-ebitda            EV / EBITDA
  fcf-yield            FCF Yield floor
  rim                  Residual Income Model (EBO)
  roic                 ROIC Excess Return (McKinsey)
  ncav                 Net Current Asset Value (Graham)
  mean-reversion       Mean Reversion to historical multiples
  reverse-dcf / rdcf   Reverse DCF (implied growth)
  peg / fwd-peg        Forward PEG
  ev-ntm               EV / NTM Revenue
  tam                  TAM Scenario
  rule40 / r40         Rule of 40
  erg                  Earnings/Revenue/Growth
  scurve               S-Curve (Logistic) TAM
  pie                  Price Implied Expectations
  ddm                  DDM H-Model (dividend stocks)
  graham               Graham Number
  multifactor / mf     Multi-Factor Price Target
  bayesian             Bayesian Ensemble (requires all others first)
  all                  Run every applicable model

Optional flags
--------------
  --wacc   RATE   Override WACC, e.g. 0.09 for 9%
  --growth RATE   Override estimated growth, e.g. 0.15 for 15%
  --plot          Generate a price-history chart with the model fair value
                  overlaid as a horizontal reference line (requires matplotlib)
  --days   N      Number of trading days of history to show in the plot
                  (default: 1000 ≈ 4 years)

Examples
--------
  python run_model.py NVDA dcf
  python run_model.py AAPL all
  python run_model.py MSFT monte-carlo --wacc 0.09
  python run_model.py GOOG peg --growth 0.18
  python run_model.py NVDA dcf --plot
  python run_model.py AAPL all --plot --days 500
"""

import sys
import os
import argparse
import math
import csv
import io
import datetime
import json
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Library imports ───────────────────────────────────────────────────────────
from valuation_models import (
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM, TERMINAL_GROWTH_RATE,
    calculate_wacc, score_model_applicability, assess_reliability,
    run_dcf, run_pfcf, run_pe, run_ev_ebitda,
    run_three_stage_dcf, run_monte_carlo_dcf, run_fcf_yield,
    run_rim, run_roic_excess_return, run_ncav, run_mean_reversion,
    run_reverse_dcf, run_forward_peg, run_ev_ntm_revenue,
    run_tam_scenario, run_rule_of_40, run_erg_valuation,
    run_scurve_tam, run_pie, run_ddm_hmodel,
    run_graham_number, run_multifactor_price_target, run_bayesian_ensemble,
    calibrate_erg_multiple,
)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# ── Model alias table ─────────────────────────────────────────────────────────
ALIASES = {
    "dcf":            "dcf",
    "three-stage":    "three-stage-dcf",
    "3dcf":           "three-stage-dcf",
    "three-stage-dcf":"three-stage-dcf",
    "monte-carlo":    "monte-carlo",
    "mc":             "monte-carlo",
    "pfcf":           "pfcf",
    "pe":             "pe",
    "ev-ebitda":      "ev-ebitda",
    "evebitda":       "ev-ebitda",
    "fcf-yield":      "fcf-yield",
    "fcfyield":       "fcf-yield",
    "rim":            "rim",
    "roic":           "roic",
    "roic-excess":    "roic",
    "ncav":           "ncav",
    "mean-reversion": "mean-reversion",
    "mean":           "mean-reversion",
    "reverse-dcf":    "reverse-dcf",
    "rdcf":           "reverse-dcf",
    "peg":            "peg",
    "fwd-peg":        "peg",
    "ev-ntm":         "ev-ntm",
    "evntm":          "ev-ntm",
    "tam":            "tam",
    "rule40":         "rule40",
    "r40":            "rule40",
    "erg":            "erg",
    "scurve":         "scurve",
    "s-curve":        "scurve",
    "pie":            "pie",
    "ddm":            "ddm",
    "graham":         "graham",
    "multifactor":    "multifactor",
    "mf":             "multifactor",
    "bayesian":       "bayesian",
    "all":            "all",
}

ALL_MODELS = [
    "dcf", "three-stage-dcf", "monte-carlo", "pfcf", "pe", "ev-ebitda",
    "fcf-yield", "rim", "roic", "ncav", "mean-reversion",
    "reverse-dcf", "peg", "ev-ntm", "tam", "rule40", "erg",
    "scurve", "pie", "ddm", "graham", "multifactor",
]


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, wacc_override=None, growth_override=None) -> dict:
    """Build the canonical d-dict from yfinance."""
    print(f"  Fetching {ticker} from yfinance...", end=" ", flush=True)
    tk = yf.Ticker(ticker)

    def _first(df, keys, default=None):
        if df is None or df.empty:
            return default
        for k in keys:
            if k in df.index:
                row = df.loc[k].dropna()
                if len(row) > 0:
                    try:
                        return float(row.iloc[0])
                    except (TypeError, ValueError):
                        pass
        return default

    # ── Info
    try:
        info = tk.info
    except Exception:
        info = {}

    def _i(key, default=None):
        v = info.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    price      = _i("currentPrice") or _i("regularMarketPrice") or _i("previousClose", 0)
    market_cap = _i("marketCap", 0)
    shares     = _i("sharesOutstanding") or _i("impliedSharesOutstanding")
    beta       = _i("beta", 1.0) or 1.0
    eps        = _i("trailingEps")
    fwd_eps    = _i("forwardEps")
    pe_ttm     = _i("trailingPE")
    div_rate   = _i("dividendRate", 0) or 0
    book_value_ps = _i("bookValue")
    sector     = info.get("sector")
    industry   = info.get("industry")

    # ── Financial statements
    try:
        inc  = tk.income_stmt
        cf   = tk.cash_flow
        bs   = tk.balance_sheet
    except Exception:
        inc = cf = bs = None

    revenue    = _first(inc, ["Total Revenue"])
    net_income = _first(inc, ["Net Income", "Net Income Common Stockholders"])
    ebitda     = _first(inc, ["EBITDA", "Normalized EBITDA"]) or _i("ebitda")
    op_margin  = _first(inc, ["Operating Income"])
    if op_margin and revenue and revenue > 0:
        op_margin = op_margin / revenue * 100
    else:
        op_margin = _i("operatingMargins", 0) * 100 if _i("operatingMargins") else None

    gross_margin = _i("grossMargins", 0) * 100 if _i("grossMargins") else None

    fcf_stmt   = _first(cf, ["Free Cash Flow"])
    op_cf      = _first(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex      = _first(cf, ["Capital Expenditure", "Capital Expenditures"])
    if fcf_stmt is not None:
        fcf = fcf_stmt
    elif op_cf is not None and capex is not None:
        fcf = op_cf + capex   # capex is typically negative
    else:
        fcf = None

    total_debt = _first(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
    if total_debt is None:
        total_debt = (_i("totalDebt") or 0)

    cash       = _first(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    if cash is None:
        cash = _i("totalCash") or 0

    cur_assets = _first(bs, ["Current Assets", "Total Current Assets"])
    tot_liab   = _first(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
    equity     = _first(bs, ["Stockholders Equity", "Common Stock Equity", "CommonStockEquity",
                              "Total Equity Gross Minority Interest"])

    print("OK")

    # ── Growth estimate
    fcf_per_share = (fcf / shares) if (fcf and shares and shares > 0) else None
    fcf_margin_v  = (fcf / revenue) if (fcf and revenue and revenue > 0) else None

    if growth_override is not None:
        growth      = growth_override
        growth_src  = "CLI override"
    elif fwd_eps and eps and eps > 0 and fwd_eps > eps:
        growth      = max(0.02, min((fwd_eps / eps) - 1.0, 0.80))
        growth_src  = f"implied from fwd EPS ({fwd_eps:.2f} / {eps:.2f})"
    elif fcf_margin_v and fcf_margin_v > 0.40:
        growth = 0.15; growth_src = "FCF margin proxy (>40%)"
    elif fcf_margin_v and fcf_margin_v > 0.20:
        growth = 0.12; growth_src = "FCF margin proxy (>20%)"
    elif fcf_margin_v and fcf_margin_v > 0.10:
        growth = 0.09; growth_src = "FCF margin proxy (>10%)"
    else:
        growth = 0.06; growth_src = "FCF margin proxy (default)"

    # ── WACC inputs
    wacc_raw = {
        "beta_yf":          beta,
        "interest_expense": _first(inc, ["Interest Expense"]),
        "income_tax_expense": _first(inc, ["Tax Provision", "Income Tax Expense"]),
        "pretax_income":    _first(inc, ["Pretax Income"]),
        "total_debt_yf":    total_debt,
    }

    # ── ROIC
    roic = None
    invested_capital = None
    if equity is not None and total_debt is not None and net_income is not None:
        invested_capital = equity + total_debt - cash
        if invested_capital and invested_capital > 0:
            roic = net_income / invested_capital

    # ── Benchmarks (simple market defaults — no TV dependency)
    benchmarks = {"pe": 22.0, "pfcf": 22.0, "ev_ebitda": 14.0,
                  "sector_name": "Market default", "peer_count": 0}

    # ── Assemble canonical d-dict
    d = {
        "ticker":           ticker.upper(),
        "price":            price,
        "market_cap":       market_cap,
        "fcf":              fcf,
        "fcf_per_share":    fcf_per_share,
        "fcf_margin":       fcf_margin_v,
        "cash":             cash,
        "total_debt":       total_debt,
        "shares":           shares,
        "eps":              eps,
        "fwd_eps":          fwd_eps,
        "fwd_rev":          None,
        "revenue":          revenue,
        "net_income":       net_income,
        "ebitda":           ebitda,
        "op_margin":        op_margin,
        "gross_margin":     gross_margin,
        "beta":             beta,
        "est_growth":       growth,
        "growth_source":    growth_src,
        "sector":           sector,
        "industry":         industry,
        "wacc_override":    wacc_override,
        "wacc_raw":         wacc_raw,
        "ev_approx":        market_cap + total_debt - cash,
        # Extended fields
        "ext": {
            "book_value_ps":         book_value_ps,
            "stockholders_equity":   equity,
            "total_current_assets":  cur_assets,
            "total_liabilities":     tot_liab,
            "roic":                  roic,
            "invested_capital":      invested_capital,
            "dividends_per_share":   div_rate if div_rate > 0 else None,
            "hist_pe_5y":            None,
            "hist_pfcf_5y":          None,
            "hist_eveb_5y":          None,
            "earnings_surprise_pct": None,
        },
    }

    # Merge ext fields into top-level d (some models read them from d directly)
    for k, v in d["ext"].items():
        if k not in d:
            d[k] = v

    return d, benchmarks


# ── Print helpers ─────────────────────────────────────────────────────────────
W = 62
SEP = "─" * W

TRADING_DAYS_PER_YEAR = 252   # business days per year used in forward projection
CALENDAR_SLIPPAGE     = 1.6   # trading-day to calendar-day conversion factor
CALENDAR_PADDING      = 60    # extra calendar days buffer for holiday gaps

def _header(ticker, model_name, price):
    print()
    print("╔" + "═" * W + "╗")
    print("║  {:.<{w}}  ║".format(f"{ticker}  ·  {model_name}", w=W-2))
    print("║  Current price: ${:<{w}}  ║".format(f"{price:,.2f}", w=W-20))
    print("╚" + "═" * W + "╝")

def _line(label, value, note="", width=26):
    label_s = f"  {label}:"
    print(f"{label_s:<{width}} {value}  {note}".rstrip())

def _mo(v):
    return f"${v:,.2f}" if v is not None else "—"

def _upside(fv, price):
    if fv is None or price is None or price == 0:
        return ""
    pct = (fv - price) / price * 100
    sign = "+" if pct >= 0 else ""
    tag  = "UNDERVALUED" if pct > 5 else ("OVERVALUED" if pct < -5 else "FAIR")
    return f"({sign}{pct:.1f}%  {tag})"

def _section(title):
    print(f"\n  {'─'*4} {title} {'─'*(W-8-len(title))}")

def _warn(msg):
    print(f"  ⚠  {msg}")

def _skip(reason):
    print(f"\n  SKIPPED — {reason}")


# ── Per-model print functions ─────────────────────────────────────────────────
def _print_dcf(r, price):
    _section("DCF Result")
    _line("Fair value",  _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",   _mo(r["mos_value"]))
    _line("WACC",        f"{r['wacc']*100:.2f}%", r.get("wacc_source",""))
    _line("Growth used", f"{r['growth']*100:.1f}%")
    det = r.get("details", {})
    if det:
        _line("PV Stage 1",  _mo(det.get("sum_pv")))
        _line("PV Terminal", _mo(det.get("pv_term")))
        pct = det.get("term_pct")
        if pct:
            note = "⚠  >80% of EV — terminal value dominates" if pct > 80 else ""
            _line("Terminal %",  f"{pct:.1f}%", note)
    if r.get("warning"):
        _warn(r["warning"])

def _print_pfcf(r, price):
    _section("P/FCF Result")
    _line("Fair value",    _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",     _mo(r["mos_value"]))
    _line("FCF/share",     _mo(r.get("fcf_per_share")))
    _line("Sector P/FCF",  f"{r.get('sector_pfcf','?')}x")
    _line("Applied mult",  f"{r.get('applied_mult','?'):.1f}x" if r.get("applied_mult") else "—")

def _print_pe(r, price):
    _section("P/E Result")
    _line("Fair value",  _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",   _mo(r["mos_value"]))
    _line("EPS (TTM)",   _mo(r.get("eps")))
    _line("Sector P/E",  f"{r.get('sector_pe','?')}x")
    _line("Applied mult",f"{r.get('applied_mult','?'):.1f}x" if r.get("applied_mult") else "—")

def _print_ev_ebitda(r, price):
    _section("EV/EBITDA Result")
    _line("Fair value",     _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",      _mo(r["mos_value"]))
    _line("EBITDA/share",   _mo(r.get("ebitda_per_share")))
    _line("Sector EV/EBIT", f"{r.get('sector_ev_ebitda','?')}x")

def _print_three_stage(r, price):
    _section("Three-Stage DCF Result")
    _line("Fair value",   _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",    _mo(r["mos_value"]))
    _line("WACC",         f"{r['wacc']*100:.2f}%")
    _line("Growth used",  f"{r['growth']*100:.1f}%")
    det = r.get("details", {})
    if det:
        _line("PV Stage 1",   _mo(det.get("sum_pv")))
        _line("PV Terminal",  _mo(det.get("pv_term")))
        pct = det.get("term_pct")
        if pct:
            _line("Terminal %",   f"{pct:.1f}%")
    if r.get("warning"):
        _warn(r["warning"])

def _print_monte_carlo(r, price):
    _section("Monte Carlo DCF Result")
    _line("Fair value (mean)", _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",         _mo(r["mos_value"]))
    _line("p10 / p25",         f"{_mo(r.get('p10'))} / {_mo(r.get('p25'))}")
    _line("p75 / p90",         f"{_mo(r.get('p75'))} / {_mo(r.get('p90'))}")
    _line("Simulations",       str(r.get("n_sims", 5000)))
    if r.get("warning"):
        _warn(r["warning"])

def _print_fcf_yield(r, price):
    _section("FCF Yield Result")
    _line("Fair value",    _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",     _mo(r["mos_value"]))
    _line("FCF/share",     _mo(r.get("fcf_per_share")))
    _line("Target yield",  f"{r.get('target_yield','?')}%")
    _line("Current yield", f"{r.get('current_yield','?')}%")
    if r.get("warning"):
        _warn(r["warning"])

def _print_rim(r, price):
    _section("Residual Income Model (EBO) Result")
    _line("Fair value",   _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",    _mo(r["mos_value"]))
    _line("Book value/sh",_mo(r.get("book_value_ps")))
    _line("Cost of equity",f"{r.get('cost_of_equity',0)*100:.2f}%")
    _line("ROE used",     f"{r.get('roe',0)*100:.1f}%")
    if r.get("warning"):
        _warn(r["warning"])

def _print_roic(r, price):
    _section("ROIC Excess Return Result")
    _line("Fair value",    _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",     _mo(r["mos_value"]))
    _line("ROIC",          f"{r.get('roic',0)*100:.1f}%")
    _line("WACC",          f"{r.get('wacc',0)*100:.2f}%")
    _line("IC/share",      _mo(r.get("ic_per_share")))
    if r.get("warning"):
        _warn(r["warning"])

def _print_ncav(r, price):
    _section("NCAV (Graham) Result")
    _line("NCAV/share",   _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",    _mo(r["mos_value"]))
    _line("Current assets",_mo(r.get("total_current_assets")))
    _line("Total liab",    _mo(r.get("total_liabilities")))
    _line("Verdict",       r.get("graham_verdict", "—"))
    if r.get("warning"):
        _warn(r["warning"])

def _print_mean_reversion(r, price):
    _section("Mean Reversion Result")
    _line("Fair value",    _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",     _mo(r["mos_value"]))
    _line("Components",    str(r.get("n_components", 0)))
    for k in ["pe_component", "pfcf_component", "eveb_component"]:
        if r.get(k):
            _line(k.replace("_"," ").title(), _mo(r[k]))
    if r.get("warning"):
        _warn(r["warning"])

def _print_reverse_dcf(r, price):
    _section("Reverse DCF Result")
    _line("Implied growth",  f"{r.get('implied_growth',0)*100:.2f}%")
    _line("Consensus growth",f"{r.get('consensus_growth',0)*100:.2f}%")
    verdict = r.get("verdict", "")
    if verdict:
        _line("Verdict",  verdict)
    if r.get("warning"):
        _warn(r["warning"])

def _print_peg(r, price):
    _section("Forward PEG Result")
    _line("Fair value",   _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",    _mo(r["mos_value"]))
    _line("Fwd EPS",      _mo(r.get("fwd_eps")))
    _line("Growth used",  f"{r.get('growth',0)*100:.1f}%")
    _line("PEG target",   str(r.get("peg_target", "?")))
    if r.get("warning"):
        _warn(r["warning"])

def _print_ev_ntm(r, price):
    _section("EV/NTM Revenue Result")
    _line("Fair value",    _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",     _mo(r["mos_value"]))
    _line("NTM Revenue",   _mo(r.get("ntm_revenue")))
    _line("EV mult used",  f"{r.get('ev_mult_used','?')}x")
    if r.get("warning"):
        _warn(r["warning"])

def _print_tam(r, price):
    _section("TAM Scenario Result")
    _line("Fair value (base)",  _mo(r.get("fair_value")), _upside(r.get("fair_value"), price))
    _line("Bear",               _mo(r.get("bear_value")))
    _line("Bull",               _mo(r.get("bull_value")))
    _line("TAM used",           _mo(r.get("tam")))
    if r.get("warning"):
        _warn(r["warning"])

def _print_rule40(r, price):
    _section("Rule of 40")
    _line("Score",   str(r.get("ro40","?")))
    _line("Cohort",  r.get("cohort","—"))
    _line("Rev growth",  f"{r.get('rev_growth',0):.1f}%")
    _line("FCF margin",  f"{r.get('fcf_margin',0):.1f}%")
    print("\n  Note: Rule of 40 is a quality signal, not a price target.")

def _print_erg(r, price):
    _section("ERG Valuation Result")
    _line("Fair value (base)",  _mo(r.get("fair_value")), _upside(r.get("fair_value"), price))
    _line("Bear",               _mo(r.get("bear_value")))
    _line("Bull",               _mo(r.get("bull_value")))
    _line("ERG multiple",       f"{r.get('erg_mult','?'):.3f}")
    if r.get("warning"):
        _warn(r["warning"])

def _print_scurve(r, price):
    _section("S-Curve TAM Result")
    _line("Fair value (base)",  _mo(r.get("fair_value")), _upside(r.get("fair_value"), price))
    _line("Bear",               _mo(r.get("bear_value")))
    _line("Bull",               _mo(r.get("bull_value")))
    _line("Implied TAM share",  f"{r.get('peak_share',0)*100:.1f}%")
    if r.get("warning"):
        _warn(r["warning"])

def _print_pie(r, price):
    _section("Price Implied Expectations Result")
    _line("Fair value",       _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",        _mo(r["mos_value"]))
    _line("Implied growth",   f"{r.get('implied_growth',0)*100:.2f}%")
    _line("Consensus growth", f"{r.get('consensus_growth',0)*100:.2f}%")
    _line("Verdict",          r.get("verdict","—"))
    if r.get("warning"):
        _warn(r["warning"])

def _print_ddm(r, price):
    _section("DDM H-Model Result")
    _line("Fair value",     _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",      _mo(r["mos_value"]))
    _line("Dividend (D0)",  _mo(r.get("dividends_per_share")))
    _line("High growth",    f"{r.get('gS',0):.1f}%")
    _line("Terminal growth",f"{r.get('gL',0):.1f}%")
    _line("Cost of equity", f"{r.get('cost_of_equity',0):.2f}%")
    if r.get("warning"):
        _warn(r["warning"])

def _print_graham(fv, price):
    _section("Graham Number Result")
    _line("Graham Number", _mo(fv), _upside(fv, price))
    print("  Note: √(22.5 × EPS × BookValue) — margin of safety floor only.")

def _print_multifactor(r, price):
    _section("Multi-Factor Price Target Result")
    _line("Fair value",      _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",       _mo(r["mos_value"]))
    _line("Score",           f"{r.get('score',0):.1f} / 100")
    _line("Expected return", f"{r.get('expected_return',0)*100:.1f}%")
    for factor in ["value_score","quality_score","growth_score","momentum_score"]:
        if r.get(factor) is not None:
            _line(factor.replace("_"," ").title(), f"{r[factor]:.1f} pts")
    if r.get("warning"):
        _warn(r["warning"])

def _print_bayesian(r, price):
    _section("Bayesian Ensemble Result")
    _line("Fair value",      _mo(r["fair_value"]), _upside(r["fair_value"], price))
    _line("MoS price",       _mo(r["mos_value"]))
    _line("Models used",     str(r.get("n_models", "?")))
    _line("Earnings adj",    f"{r.get('earnings_adj',0)*100:+.1f}%" if r.get("earnings_adj") else "none")
    if r.get("warning"):
        _warn(r["warning"])


# ── Run a single model ────────────────────────────────────────────────────────
def run_one(model_key, d, benchmarks):
    """Run model_key, print result. Returns result dict/float or None."""
    price = d["price"]
    ext   = d.get("ext", {})

    # ── Classic ───────────────────────────────────────────────────────────────
    if model_key == "dcf":
        if not (d["fcf"] and d["fcf"] > 0 and d["shares"]):
            _skip("no positive FCF"); return None
        r = run_dcf(d); _print_dcf(r, price); return r

    if model_key == "three-stage-dcf":
        if not (d["fcf"] and d["fcf"] > 0 and d["shares"]):
            _skip("no positive FCF"); return None
        r = run_three_stage_dcf(d)
        if r: _print_three_stage(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "monte-carlo":
        try:
            r = run_monte_carlo_dcf(d)
            if r: _print_monte_carlo(r, price)
            else: _skip("no positive FCF or numpy unavailable")
            return r
        except Exception as e:
            _skip(str(e)); return None

    if model_key == "pfcf":
        if not (d["fcf_per_share"] and d["fcf_per_share"] > 0):
            _skip("no positive FCF per share"); return None
        r = run_pfcf(d, benchmarks); _print_pfcf(r, price); return r

    if model_key == "pe":
        if not (d["eps"] and d["eps"] > 0):
            _skip("no positive EPS"); return None
        r = run_pe(d, benchmarks); _print_pe(r, price); return r

    if model_key == "ev-ebitda":
        if not (d["ebitda"] and d["ebitda"] > 0 and d["shares"]):
            _skip("no positive EBITDA"); return None
        r = run_ev_ebitda(d, benchmarks); _print_ev_ebitda(r, price); return r

    if model_key == "fcf-yield":
        if not (d["fcf_per_share"] and d["fcf_per_share"] > 0):
            _skip("no positive FCF per share"); return None
        r = run_fcf_yield(d)
        if r: _print_fcf_yield(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "rim":
        if not (ext.get("book_value_ps") and d.get("eps") and d["eps"] > 0):
            _skip("no book value per share or no positive EPS"); return None
        r = run_rim(d)
        if r: _print_rim(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "roic":
        if not (ext.get("roic") is not None and ext.get("stockholders_equity")):
            _skip("no ROIC data — need stockholders equity"); return None
        r = run_roic_excess_return(d)
        if r: _print_roic(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "ncav":
        if not (ext.get("total_current_assets") and ext.get("total_liabilities") and d.get("shares")):
            _skip("no current assets / total liabilities data"); return None
        r = run_ncav(d)
        if r: _print_ncav(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "mean-reversion":
        if not any(ext.get(k) for k in ["hist_pe_5y","hist_pfcf_5y","hist_eveb_5y"]):
            _skip("no 5-year historical multiples (yfinance doesn't provide these; use valuationMaster for full run)")
            return None
        r = run_mean_reversion(d)
        if r: _print_mean_reversion(r, price)
        else: _skip("model returned no result")
        return r

    # ── Growth ────────────────────────────────────────────────────────────────
    if model_key == "reverse-dcf":
        r = run_reverse_dcf(d)
        if r: _print_reverse_dcf(r, price)
        else: _skip("no positive FCF")
        return r

    if model_key == "peg":
        r = run_forward_peg(d)
        if r: _print_peg(r, price)
        else: _skip("no forward EPS or growth data")
        return r

    if model_key == "ev-ntm":
        r = run_ev_ntm_revenue(d)
        if r: _print_ev_ntm(r, price)
        else: _skip("no revenue data")
        return r

    if model_key == "tam":
        r = run_tam_scenario(d)
        if r: _print_tam(r, price)
        else: _skip("no revenue or gross margin data")
        return r

    if model_key == "rule40":
        r = run_rule_of_40(d)
        if r: _print_rule40(r, price)
        else: _skip("no revenue growth or FCF margin data")
        return r

    if model_key == "erg":
        try:
            erg_mult = calibrate_erg_multiple(sector=d.get("sector"))
        except Exception:
            erg_mult = None
        r = run_erg_valuation(d, erg_mult)
        if r: _print_erg(r, price)
        else: _skip("no revenue or growth data")
        return r

    if model_key == "scurve":
        r = run_scurve_tam(d)
        if r: _print_scurve(r, price)
        else: _skip("no revenue data or market share outside 0.1%–70% range")
        return r

    if model_key == "pie":
        r = run_pie(d)
        if r: _print_pie(r, price)
        else: _skip("no price / growth data")
        return r

    if model_key == "ddm":
        if not (ext.get("dividends_per_share") and ext["dividends_per_share"] > 0):
            _skip("no dividend — DDM only applies to dividend-paying stocks"); return None
        r = run_ddm_hmodel(d)
        if r: _print_ddm(r, price)
        else: _skip("model returned no result")
        return r

    if model_key == "graham":
        fv = run_graham_number(d)
        if fv and fv > 0:
            _print_graham(fv, price)
        else:
            _skip("no EPS or book value")
        return fv

    # ── Meta ──────────────────────────────────────────────────────────────────
    if model_key == "multifactor":
        r = run_multifactor_price_target(d, benchmarks)
        if r: _print_multifactor(r, price)
        else: _skip("insufficient data")
        return r

    if model_key == "bayesian":
        print("\n  Running all models first to feed Bayesian ensemble...")
        all_results = []
        for mk in ALL_MODELS:
            try:
                res = run_one(mk, d, benchmarks)
                if isinstance(res, dict) and res.get("fair_value"):
                    all_results.append(res)
            except Exception:
                pass
        if len(all_results) < 2:
            _skip("need at least 2 valid model results"); return None
        appl = {r["method"]: score_model_applicability(r["method"], r, d, []) for r in all_results}
        earnings_surp = ext.get("earnings_surprise_pct")
        r = run_bayesian_ensemble(d, all_results, appl, earnings_surp)
        if r: _print_bayesian(r, price)
        else: _skip("model returned no result")
        return r

    print(f"\n  Unknown model key: {model_key}")
    return None


# ── Backtest helpers (ported from valuationMaster.py) ─────────────────────────

_REQ_HEADERS = {
    "User-Agent": "ValuationScript/1.0 research@example.com",
    "Accept":     "*/*",
}

def _http_get(url: str, timeout: int = 20) -> bytes:
    try:
        req = urllib.request.Request(url, headers=_REQ_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return b""

def _edgar_cik(ticker: str) -> str:
    raw = _http_get("https://www.sec.gov/files/company_tickers.json", timeout=15)
    if raw:
        try:
            for entry in json.loads(raw).values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception:
            pass
    return ""

def _edgar_xbrl_facts(cik: str) -> dict:
    raw = _http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=30)
    if not raw:
        return {}
    try:
        return json.loads(raw).get("facts", {})
    except Exception:
        return {}

def _xbrl_annual_series(facts: dict, *concepts) -> list:
    for concept in concepts:
        for ns in ("us-gaap", "dei"):
            try:
                units    = facts[ns][concept]["units"]
                unit_key = "USD" if "USD" in units else list(units.keys())[0]
                annual   = {}
                for e in units[unit_key]:
                    if e.get("form") not in ("10-K", "10-K/A"):
                        continue
                    end, val = e.get("end", ""), e.get("val")
                    if end and val is not None:
                        ex = annual.get(end)
                        if ex is None or e.get("filed", "") > ex.get("filed", ""):
                            annual[end] = {"date": end, "value": float(val),
                                           "filed": e.get("filed", "")}
                if annual:
                    return sorted(annual.values(), key=lambda x: x["date"])
            except (KeyError, TypeError, IndexError):
                continue
    return []

def _fetch_prices(ticker: str, days: int) -> list:
    """Fetch daily closes via yfinance (primary) with Stooq fallback.
    Returns [{date, close}, ...] oldest-first."""
    import math

    cal_days = math.ceil(days * CALENDAR_SLIPPAGE) + CALENDAR_PADDING

    # ── Primary: yfinance — fast, reliable, no per-variant probe loop ────────
    # Use explicit start/end dates rather than period="NNNd": Yahoo Finance only
    # officially supports specific period strings ("1d","5d","1mo"…"max"), so
    # large day values like "1660d" may silently return empty data in some
    # yfinance versions.
    try:
        import yfinance as yf
        _end   = datetime.date.today()
        _start = _end - datetime.timedelta(days=cal_days)
        hist = yf.Ticker(ticker).history(
            start=_start.strftime("%Y-%m-%d"),
            end=_end.strftime("%Y-%m-%d"),
        )
        if hist is not None and not hist.empty:
            pts = []
            for ts, row in hist.iterrows():
                try:
                    cl = float(row["Close"])
                    dt_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                    if cl > 0 and dt_str:
                        pts.append({"date": dt_str, "close": cl})
                except (ValueError, TypeError):
                    continue
            pts.sort(key=lambda x: x["date"])
            pts = pts[-days:]
            if len(pts) >= 10:
                print(f"  [yfinance] {len(pts)} price points for '{ticker}'")
                return pts
    except Exception as e:
        print(f"  [yfinance] price error: {e}")

    # ── Fallback: Stooq ───────────────────────────────────────────────────────
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=cal_days)
    df, dt   = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
    t        = ticker.lower()
    for sym in list(dict.fromkeys([t+".us", t+".ca", t+".uk", t+".de", t])):
        url = f"https://stooq.com/q/d/l/?s={sym}&d1={df}&d2={dt}&i=d"
        raw = _http_get(url)
        if not raw or b"No data" in raw or b"Exceeded" in raw or len(raw) < 50:
            continue
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))))
            pts  = [{"date": (r.get("Date") or r.get("date","")).strip(),
                     "close": float(r.get("Close") or r.get("close") or 0)}
                    for r in rows if float(r.get("Close") or r.get("close") or 0) > 0]
            if len(pts) < 10:
                continue
            pts.sort(key=lambda x: x["date"])
            print(f"  [Stooq] {len(pts)} price points for '{sym}'")
            return pts[-days:]
        except Exception:
            continue

    print(f"  [prices] No data available for '{ticker}'")
    return []

def _fetch_fundamentals(ticker: str) -> list:
    """Fetch annual 10-K snapshots from SEC EDGAR XBRL. Returns list oldest-first."""
    print(f"  [EDGAR] Looking up CIK for {ticker}…", end=" ", flush=True)
    cik = _edgar_cik(ticker)
    if not cik:
        print("not found"); return []
    print(f"CIK {cik}. Fetching XBRL…", end=" ", flush=True)
    facts = _edgar_xbrl_facts(cik)
    if not facts:
        print("no data"); return []

    rev_s  = _xbrl_annual_series(facts, "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "SalesRevenueGoodsNet")
    ni_s   = _xbrl_annual_series(facts, "NetIncomeLoss", "NetIncome", "ProfitLoss")
    opcf_s = _xbrl_annual_series(facts,
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    capx_s = _xbrl_annual_series(facts,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements")
    debt_s = _xbrl_annual_series(facts,
        "LongTermDebtNoncurrent", "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations")
    gp_s   = _xbrl_annual_series(facts, "GrossProfit")
    eb_s   = _xbrl_annual_series(facts,
        "OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes")

    def _d(s): return {e["date"]: e["value"] for e in s}
    rev_d, ni_d, opcf_d = _d(rev_s), _d(ni_s), _d(opcf_s)
    capx_d, debt_d      = _d(capx_s), _d(debt_s)
    gp_d, eb_d          = _d(gp_s), _d(eb_s)

    all_dates = sorted(set(list(rev_d) + list(ni_d) + list(opcf_d)))
    if not all_dates:
        print("no series"); return []

    snaps = []
    for date in all_dates:
        rev, ni    = rev_d.get(date), ni_d.get(date)
        opcf, capx = opcf_d.get(date), capx_d.get(date)
        debt, gp   = debt_d.get(date), gp_d.get(date)
        ebit       = eb_d.get(date)
        fcf        = (opcf - capx) if (opcf and capx) else (opcf * 0.85 if opcf else None)
        gm         = (gp / rev * 100)   if (gp   and rev and rev > 0) else None
        op_mar     = (ebit / rev * 100) if (ebit and rev and rev > 0) else None
        snaps.append({"date": date, "revenue": rev, "net_income": ni, "fcf": fcf,
                      "total_debt": debt, "gross_margin": gm, "op_margin": op_mar})

    print(f"{len(snaps)} snapshots ({snaps[0]['date']} → {snaps[-1]['date']})")
    return snaps

def _pick_snapshot(snaps: list, target: str) -> dict:
    """Return most recent snapshot whose 10-K was publicly available by `target` date
    (75-day filing lag after fiscal year end)."""
    best = None
    for s in snaps:
        try:
            avail = (datetime.datetime.strptime(s["date"], "%Y-%m-%d")
                     + datetime.timedelta(days=75)).strftime("%Y-%m-%d")
        except ValueError:
            continue
        if avail <= target:
            best = s
    return best

def _rebuild_snapshot(d_cur: dict, snap: dict, hist_price: float, bm: dict) -> dict:
    """Overlay historical snapshot financials onto a copy of the current d-dict."""
    d = dict(d_cur)
    shares = d_cur["shares"]

    def _or(a, b):
        return a if (a is not None and a == a) else b

    rev    = _or(snap.get("revenue"),    d_cur["revenue"])
    ni     = _or(snap.get("net_income"), d_cur["net_income"])
    fcf    = _or(snap.get("fcf"),        d_cur["fcf"])
    debt   = _or(snap.get("total_debt"), d_cur["total_debt"])
    gm     = _or(snap.get("gross_margin"), d_cur["gross_margin"])
    op_mar = _or(snap.get("op_margin"),  d_cur["op_margin"])

    eps    = (ni  / shares) if (ni  and shares and shares > 0) else d_cur["eps"]
    fcf_ps = (fcf / shares) if (fcf and shares and shares > 0) else d_cur["fcf_per_share"]
    fcf_mr = (fcf / rev)    if (fcf and rev    and rev    > 0) else d_cur["fcf_margin"]

    ebitda = None
    if rev and op_mar:   ebitda = rev * (op_mar / 100) + rev * 0.05
    elif ni:             ebitda = ni * 1.35

    if   fcf_mr and fcf_mr > 0.40: growth = 0.15
    elif fcf_mr and fcf_mr > 0.20: growth = 0.12
    elif fcf_mr and fcf_mr > 0.10: growth = 0.09
    else:                           growth = 0.06

    mktcap    = hist_price * shares if shares else d_cur["market_cap"]
    ev_approx = mktcap + debt

    d.update({
        "price": hist_price, "market_cap": mktcap, "revenue": rev, "net_income": ni,
        "fcf": fcf, "fcf_per_share": fcf_ps, "fcf_margin": fcf_mr,
        "total_debt": debt, "cash": d_cur["cash"],
        "ebitda": ebitda, "gross_margin": gm, "op_margin": op_mar,
        "eps": eps, "shares": shares, "est_growth": growth,
        "growth_source": "FCF-margin proxy (historical)", "ev_approx": ev_approx,
        "peg": ((hist_price / eps) / (growth * 100)) if (eps and eps > 0) else None,
        "current_pe": (hist_price / eps) if (eps and eps > 0) else None,
        "current_pfcf": (mktcap / fcf) if (fcf and fcf > 0) else None,
        "fwd_eps": None, "fwd_rev": None,
        "rev_growth_pct": None, "eps_growth_pct": None,
        "wacc_override": d_cur.get("wacc_override"),
        "wacc_raw":      d_cur.get("wacc_raw", {}),
    })
    return d

# model-key → canonical method name used in _run_methods_for_snapshot result dict
_KEY_TO_METHOD = {
    "dcf":            "DCF",            "three-stage-dcf": "Three-Stage DCF",
    "monte-carlo":    "Monte Carlo DCF","pfcf":            "P/FCF",
    "pe":             "P/E",            "ev-ebitda":       "EV/EBITDA",
    "fcf-yield":      "FCF Yield",      "rim":             "RIM",
    "roic":           "ROIC Excess Return", "ncav":        "NCAV",
    "reverse-dcf":    "Reverse DCF",    "peg":             "Fwd PEG",
    "ev-ntm":         "EV/NTM Rev",     "tam":             "TAM Scenario",
    "rule40":         "Rule of 40",     "erg":             "ERG",
    "scurve":         "S-Curve TAM",    "pie":             "PIE",
    "ddm":            "DDM",            "graham":          "Graham Number",
    "multifactor":    "Multi-Factor",
}

def _run_one_snap(model_key: str, d_h: dict, bm: dict) -> float:
    """Run a single model on a historical d-dict. Returns fair_value or None."""
    try:
        if model_key == "dcf":
            if d_h.get("fcf") and d_h["fcf"] > 0 and d_h.get("shares"):
                r = run_dcf(d_h); return r["fair_value"] if r else None
        elif model_key == "three-stage-dcf":
            if d_h.get("fcf") and d_h["fcf"] > 0 and d_h.get("shares"):
                r = run_three_stage_dcf(d_h); return r["fair_value"] if r else None
        elif model_key == "monte-carlo":
            if d_h.get("fcf") and d_h["fcf"] > 0 and d_h.get("shares"):
                r = run_monte_carlo_dcf(d_h, n_sims=500); return r["fair_value"] if r else None
        elif model_key == "pfcf":
            if d_h.get("fcf_per_share") and d_h["fcf_per_share"] > 0:
                r = run_pfcf(d_h, bm); return r["fair_value"] if r else None
        elif model_key == "pe":
            if d_h.get("eps") and d_h["eps"] > 0:
                r = run_pe(d_h, bm); return r["fair_value"] if r else None
        elif model_key == "ev-ebitda":
            if d_h.get("ebitda") and d_h["ebitda"] > 0 and d_h.get("shares"):
                r = run_ev_ebitda(d_h, bm); return r["fair_value"] if r else None
        elif model_key == "fcf-yield":
            if d_h.get("fcf_per_share") and d_h["fcf_per_share"] > 0:
                r = run_fcf_yield(d_h); return r["fair_value"] if r else None
        elif model_key == "rim":
            ext = d_h.get("ext") or {}
            if ext.get("book_value_ps") and d_h.get("eps") and d_h["eps"] > 0:
                r = run_rim(d_h); return r["fair_value"] if r else None
        elif model_key == "roic":
            ext = d_h.get("ext") or {}
            if ext.get("roic") and ext.get("stockholders_equity"):
                r = run_roic_excess_return(d_h); return r["fair_value"] if r else None
        elif model_key == "ncav":
            ext = d_h.get("ext") or {}
            if ext.get("total_current_assets") is not None and ext.get("total_liabilities") is not None:
                r = run_ncav(d_h); return r["fair_value"] if r else None
        elif model_key == "reverse-dcf":
            r = run_reverse_dcf(d_h)
            if r:
                for s in r.get("scenarios", []):
                    if s.get("label") == "Street" and s.get("fv", 0) > 0:
                        return s["fv"]
        elif model_key == "peg":
            r = run_forward_peg(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "ev-ntm":
            r = run_ev_ntm_revenue(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "tam":
            r = run_tam_scenario(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "rule40":
            r = run_rule_of_40(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "erg":
            mult = calibrate_erg_multiple(sector=d_h.get("sector"))
            r = run_erg_valuation(d_h, mult); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "scurve":
            r = run_scurve_tam(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "pie":
            r = run_pie(d_h); return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "ddm":
            ext = d_h.get("ext") or {}
            if ext.get("dividends_per_share") and ext["dividends_per_share"] > 0:
                r = run_ddm_hmodel(d_h); return r["fair_value"] if r else None
        elif model_key == "graham":
            fv = run_graham_number(d_h); return fv if (fv and fv > 0) else None
        elif model_key == "multifactor":
            r = run_multifactor_price_target(d_h, bm)
            return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "mean-reversion":
            r = run_mean_reversion(d_h)
            return r["fair_value"] if (r and r.get("fair_value", 0) > 0) else None
        elif model_key == "bayesian":
            # For historical snapshots: run all models, return score-weighted mean
            # (mirrors the full Bayesian ensemble without requiring live applicability data)
            _snap_fvs = _run_all_snap(d_h, bm)
            _valid = [v for v in _snap_fvs.values() if v and v > 0]
            # Trim outliers: discard values beyond 3× median to avoid TAM/ERG distortion
            if len(_valid) >= 3:
                _med = sorted(_valid)[len(_valid) // 2]
                _valid = [v for v in _valid if v <= _med * 3.0]
            return (sum(_valid) / len(_valid)) if len(_valid) >= 3 else None
    except Exception:
        pass
    return None

def _run_all_snap(d_h: dict, bm: dict) -> dict:
    """Run all applicable models on a historical snapshot.
    Returns {method_name: fair_value} using the same keys as _KEY_TO_METHOD values."""
    results = {}
    for key in [k for k in ALL_MODELS if k not in ("bayesian",)]:
        fv = _run_one_snap(key, d_h, bm)
        if fv and fv > 0:
            results[_KEY_TO_METHOD.get(key, key.upper())] = fv
    return results


# ── Backtest + plot ────────────────────────────────────────────────────────────
def plot_backtest(ticker: str, d: dict, benchmarks: dict,
                  model_key: str, days: int, label: str):
    """
    True rolling backtest: re-run the model at each annual fundamental snapshot
    over the past `days` trading days, producing an evolving fair-value curve.
    Plots the result with a polished dark-theme chart.
    """
    try:
        import matplotlib
        # Force non-interactive backend before importing pyplot.
        # When launched as a subprocess by the Flask launcher (start_new_session=True)
        # the process is detached from macOS's window server, so the default MacOSX
        # backend fails silently — plt.savefig() produces nothing and the plot is lost.
        # Agg renders to a file buffer with no display dependency.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
        import matplotlib.patches as mpatches
    except ImportError:
        print("\n  ERROR: matplotlib is not installed.  pip install matplotlib")
        return

    # ── Body (all exceptions surfaced explicitly — no silent crashes) ────────
    try:
        _plot_backtest_body(ticker, d, benchmarks, model_key, days, label, plt,
                            mdates, mticker, mpatches)
    except Exception as _plot_err:
        import traceback
        print(f"\n  [Plot] ERROR: {_plot_err}")
        traceback.print_exc()


def _plot_backtest_body(ticker, d, benchmarks, model_key, days, label,
                        plt, mdates, mticker, mpatches):
    # ── 1. Fetch price history ────────────────────────────────────────────────
    print(f"\n[Plot] Fetching {days}-day price history…")
    prices = _fetch_prices(ticker, days)
    if not prices:
        print("  Cannot build plot — no price history available.")
        return

    # ── 2. Fetch annual fundamental snapshots ────────────────────────────────
    print("[Plot] Fetching fundamental snapshots from SEC EDGAR…")
    snaps = _fetch_fundamentals(ticker)
    if not snaps:
        print("  No EDGAR data — plotting price history only (no fair-value curve).")

    # ── 3. Build time series ──────────────────────────────────────────────────
    # Cache computed fair values by snapshot date so each snap is only run once.
    snap_cache: dict = {}
    is_all = (model_key == "all")

    time_series = []          # [{date, price, fv or fvs}]
    snap_change_dates = []    # dates where the active snapshot switches (for vertical markers)
    prev_snap_date = None

    for pt in prices:
        date_str  = pt["date"]
        hist_price = pt["close"]
        snap = _pick_snapshot(snaps, date_str) if snaps else None

        snap_date = snap["date"] if snap else None
        if snap_date != prev_snap_date:
            if prev_snap_date is not None:
                snap_change_dates.append(date_str)
            prev_snap_date = snap_date

        if snap and snap_date not in snap_cache:
            d_h = _rebuild_snapshot(d, snap, hist_price, benchmarks)
            if is_all:
                snap_cache[snap_date] = _run_all_snap(d_h, benchmarks)
            else:
                fv = _run_one_snap(model_key, d_h, benchmarks)
                snap_cache[snap_date] = fv
        elif snap is None:
            snap_cache[None] = None

        cached = snap_cache.get(snap_date)
        time_series.append({
            "date":  datetime.datetime.strptime(date_str, "%Y-%m-%d"),
            "price": hist_price,
            "fv":    cached,     # float (single) or dict (all) or None
        })

    if not time_series:
        print("  No time-series data to plot.")
        return

    dates_dt = [p["date"] for p in time_series]
    closes   = [p["price"] for p in time_series]

    # ── 4. Compute MAPE / accuracy per model ──────────────────────────────────
    def _mape_stats(fv_series, price_series):
        errs = [abs(fv - pr) / pr * 100
                for fv, pr in zip(fv_series, price_series)
                if fv and fv > 0 and pr > 0]
        if not errs:
            return None, None
        mape = sum(errs) / len(errs)
        return round(mape, 1), round(100 - mape, 1)

    if is_all:
        # Collect all method names that have at least one data point
        all_methods = sorted({m for pt in time_series
                               if isinstance(pt["fv"], dict)
                               for m in pt["fv"]})
        method_fvs = {
            m: [pt["fv"].get(m) if isinstance(pt["fv"], dict) else None
                for pt in time_series]
            for m in all_methods
        }
        method_stats_map = {
            m: _mape_stats(method_fvs[m], closes) for m in all_methods
        }
    else:
        fv_series = [pt["fv"] for pt in time_series]
        mape, acc = _mape_stats(fv_series, closes)

    # ── 5. Forward projection (1 year of business days beyond last price) ────────
    _last_dt = dates_dt[-1]
    future_dates: list = []
    _fd = _last_dt
    for _ in range(TRADING_DAYS_PER_YEAR):
        _fd += datetime.timedelta(days=1)
        while _fd.weekday() >= 5:        # skip Saturday / Sunday
            _fd += datetime.timedelta(days=1)
        future_dates.append(_fd)

    # Run model(s) on the live d-dict for the forward projection extension
    if is_all:
        _current_fvs: dict = {}
        for _key in [k for k in ALL_MODELS if k not in ("bayesian",)]:
            _fv = _run_one_snap(_key, d, benchmarks)
            if _fv and _fv > 0:
                _current_fvs[_KEY_TO_METHOD.get(_key, _key.upper())] = _fv
    else:
        _current_fv_proj = _run_one_snap(model_key, d, benchmarks)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    BG           = "#0e1117"
    PRICE_CLR    = "#ffffff"
    PRED_CLR     = "#4f8ef7"
    MODEL_COLORS = [
        "#00c896", "#bf6ff0", "#f472b6", "#34d399", "#fb923c",
        "#facc15", "#38bdf8", "#ff6b9d", "#27ae60", "#f59e0b",
        "#e87c3e", "#a855f7", "#10b981", "#94a3b8", "#e05c5c",
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Prediction zone shading (right of today) ──────────────────────────────
    if future_dates:
        ax.axvspan(_last_dt, future_dates[-1],
                   alpha=0.045, color=PRED_CLR, zorder=0, label="_nolegend_")
        ax.axvline(_last_dt, color="#555577", linewidth=0.9,
                   linestyle="--", alpha=0.55, zorder=7, label="_nolegend_")
        _mid_future = future_dates[len(future_dates) // 2]
        ax.text(_mid_future, 0.985, "PROJECTION",
                transform=ax.get_xaxis_transform(),
                color="#555577", fontsize=8, va="top", ha="center",
                fontfamily="monospace")

    # ── Price line ────────────────────────────────────────────────────────────
    ax.plot(dates_dt, closes, color=PRICE_CLR, linewidth=2.2,
            label=f"{ticker}  (price)", zorder=5)
    ax.fill_between(dates_dt, closes, alpha=0.05, color=PRICE_CLR, zorder=2)

    if is_all:
        # ── ALL mode: solid historical line + dashed projection per model ────
        for ci, method in enumerate(all_methods):
            col = MODEL_COLORS[ci % len(MODEL_COLORS)]
            fvs = method_fvs[method]
            # Draw contiguous solid segments (historical data only)
            seg_x, seg_y = [], []
            for dt, fv in zip(dates_dt, fvs):
                if fv and fv > 0:
                    seg_x.append(dt); seg_y.append(fv)
                else:
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=col, linewidth=1.4,
                                linestyle="-", alpha=0.78, zorder=4)
                        seg_x, seg_y = [], []
            if seg_x:
                mape_m, acc_m = method_stats_map[method]
                lbl = (f"{method}  ${seg_y[-1]:,.0f}"
                       + (f"  |  MAPE {mape_m:.0f}%" if mape_m else ""))
                ax.plot(seg_x, seg_y, color=col, linewidth=1.4,
                        linestyle="-", alpha=0.78, zorder=4, label=lbl)
                # Dashed extension into projection zone
                _fv_p = _current_fvs.get(method)
                if future_dates and _fv_p and _fv_p > 0:
                    ax.plot([_last_dt] + future_dates,
                            [_fv_p] * (1 + len(future_dates)),
                            color=col, linewidth=0.9, linestyle="--",
                            alpha=0.40, zorder=4)
    else:
        # ── Single model: green/red fill + solid historical + dashed projection
        fv_clean   = [fv if (fv and fv > 0) else None for fv in fv_series]
        valid_mask = [fv is not None for fv in fv_clean]
        fv_fill    = [fv if fv is not None else 0.0 for fv in fv_clean]

        # Green fill where FV > price (undervalued zone)
        ax.fill_between(dates_dt, fv_fill, closes,
                        where=[m and fv >= pr
                               for m, fv, pr in zip(valid_mask, fv_fill, closes)],
                        alpha=0.12, color="#00c896", zorder=2, interpolate=True,
                        label="_nolegend_")
        # Red fill where FV < price (overvalued zone)
        ax.fill_between(dates_dt, fv_fill, closes,
                        where=[m and fv < pr
                               for m, fv, pr in zip(valid_mask, fv_fill, closes)],
                        alpha=0.09, color="#e05c5c", zorder=2, interpolate=True,
                        label="_nolegend_")

        # Solid historical fair-value line (steps at each new 10-K snapshot)
        seg_x, seg_y = [], []
        for dt, fv in zip(dates_dt, fv_clean):
            if fv is not None:
                seg_x.append(dt); seg_y.append(fv)
            elif seg_x:
                ax.plot(seg_x, seg_y, color=MODEL_COLORS[0], linewidth=2.0,
                        linestyle="-", alpha=0.90, zorder=6)
                seg_x, seg_y = [], []
        if seg_x:
            lbl = f"{label}  ${seg_y[-1]:,.2f}"
            if mape:
                lbl += f"  |  MAPE {mape:.0f}%  acc {acc:.0f}/100"
            ax.plot(seg_x, seg_y, color=MODEL_COLORS[0], linewidth=2.0,
                    linestyle="-", alpha=0.90, zorder=6, label=lbl)

        # Dashed projection extension (current model on live data)
        if future_dates and _current_fv_proj and _current_fv_proj > 0:
            ax.plot([_last_dt] + future_dates,
                    [_current_fv_proj] * (1 + len(future_dates)),
                    color=MODEL_COLORS[0], linewidth=1.6, linestyle="--",
                    alpha=0.55, zorder=6,
                    label=f"projection  ${_current_fv_proj:,.2f}")

    # ── Snapshot-change vertical markers ─────────────────────────────────────
    first_marker = True
    for sc_date in snap_change_dates:
        try:
            sc_dt = datetime.datetime.strptime(sc_date, "%Y-%m-%d")
            kw = {"label": "New 10-K filing"} if first_marker else {"label": "_nolegend_"}
            ax.axvline(sc_dt, color="#ffffff", linewidth=0.6, linestyle=":",
                       alpha=0.18, zorder=3, **kw)
            first_marker = False
        except ValueError:
            pass

    # ── Stats annotation (single model only) ─────────────────────────────────
    if not is_all and mape is not None:
        acc_color = "#00c896" if acc >= 75 else "#f0a500" if acc >= 50 else "#e05c5c"
        stats_txt = f"MAPE: {mape:.1f}%\nAccuracy: {acc:.0f}/100"
        ax.text(0.985, 0.97, stats_txt,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontfamily="monospace", color=acc_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0e1117",
                          edgecolor="#2a2d3e", alpha=0.85))

    # ── Axes & grid ───────────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    tick_interval = max(1, days // 250)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30, ha="right", fontsize=8, color="#888888")
    plt.yticks(fontsize=8, color="#888888")

    for spine in ax.spines.values():
        spine.set_color("#2a2d3e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#1a1c2c", linewidth=0.7, zorder=1)
    ax.grid(axis="x", color="#161828", linewidth=0.5, zorder=1)
    ax.set_xlim(dates_dt[0], future_dates[-1] if future_dates else dates_dt[-1])

    # ── Title ─────────────────────────────────────────────────────────────────
    n_snaps = len(snap_cache) - (1 if None in snap_cache else 0)
    ax.set_title(
        f"{ticker}  ·  {label}  ·  {len(prices)}-day backtest  "
        f"({n_snaps} annual snapshot{'s' if n_snaps != 1 else ''})",
        color="#cccccc", fontsize=12, pad=14, loc="left",
        fontfamily="monospace",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    n_legend_items = len(ax.get_legend_handles_labels()[1])
    leg_cols = 2 if n_legend_items > 8 else 1
    ax.legend(
        loc="upper left", fontsize=8, framealpha=0.25, ncol=leg_cols,
        facecolor=BG, edgecolor="#2a2d3e", labelcolor="#cccccc",
        handlelength=2.2,
    )

    plt.tight_layout(pad=1.5)

    # ── Save → open in default viewer (non-blocking) ──────────────────────────
    safe_label = label.lower().replace(" ", "_").replace("/", "_")
    fname = f"{ticker}_{safe_label}_{len(prices)}d_backtest.png"
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {fpath}")
    # When launched from ValuationSuite, _reader_thread detects the path above
    # and opens the file — skip opening here to avoid duplicates.
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        try:
            import platform, subprocess as _sp
            if platform.system() == "Darwin":
                _sp.Popen(["open", fpath])
            else:
                import webbrowser
                webbrowser.open(f"file://{fpath}")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Run a single valuation model on a ticker and print to terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ticker", help="Ticker symbol (e.g. NVDA)")
    p.add_argument("model",  help="Model name or alias (e.g. dcf, peg, all)")
    p.add_argument("--wacc",   type=float, default=None, help="Override WACC (e.g. 0.09 for 9%%)")
    p.add_argument("--growth", type=float, default=None, help="Override growth rate (e.g. 0.15 for 15%%)")
    p.add_argument("--plot",   action="store_true",      help="Plot historical price with fair-value overlay")
    p.add_argument("--days",   type=int,   default=1000, metavar="N",
                   help="Number of trading days of price history to show in the plot (default: 1000)")
    args = p.parse_args()

    model_input = args.model.lower().strip()
    if model_input not in ALIASES:
        print(f"ERROR: Unknown model '{args.model}'.")
        print("Run with --help for the full list of model aliases.")
        sys.exit(1)
    model_key = ALIASES[model_input]

    ticker = args.ticker.upper()
    print(f"\nFetching data for {ticker}...")
    try:
        d, benchmarks = fetch_data(ticker, args.wacc, args.growth)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Price: ${d['price']:,.2f}  |  Market cap: ${d['market_cap']/1e9:.1f}B")
    print(f"  Growth: {d['est_growth']*100:.1f}%  ({d['growth_source']})")
    if d.get("sector"):
        print(f"  Sector: {d['sector']}" + (f"  ·  {d['industry']}" if d.get("industry") else ""))

    price = d["price"]

    if model_key == "all":
        _header(ticker, "All Models", price)
        all_results = []
        for mk in ALL_MODELS:
            label = mk.upper().replace("-", " ")
            print(f"\n{'─'*4} {label} {'─'*(W-6-len(label))}")
            try:
                res = run_one(mk, d, benchmarks)
                if isinstance(res, dict) and res.get("fair_value"):
                    all_results.append(res)
            except Exception as e:
                print(f"  ERROR: {e}")

        # Bayesian last
        print(f"\n{'─'*4} BAYESIAN ENSEMBLE {'─'*(W-22)}")
        if len(all_results) >= 2:
            appl = {r["method"]: score_model_applicability(r["method"], r, d, []) for r in all_results}
            earnings_surp = d.get("ext", {}).get("earnings_surprise_pct")
            r = run_bayesian_ensemble(d, all_results, appl, earnings_surp)
            if r:
                _print_bayesian(r, price)
                all_results.append(r)
        else:
            _skip("need at least 2 valid results for Bayesian ensemble")

        # Summary table
        price_results = [r for r in all_results if isinstance(r, dict) and r.get("fair_value")]
        if price_results:
            print(f"\n{'═'*W}")
            print(f"  {'MODEL':<26} {'FAIR VALUE':>10}  {'UPSIDE':>8}  {'MoS PRICE':>10}")
            print(f"  {'─'*26} {'─'*10}  {'─'*8}  {'─'*10}")
            for r in sorted(price_results, key=lambda x: x.get("fair_value", 0), reverse=True):
                fv  = r["fair_value"]
                mos = r.get("mos_value", fv * 0.8)
                pct = (fv - price) / price * 100 if price else 0
                sign = "+" if pct >= 0 else ""
                print(f"  {r['method']:<26} {_mo(fv):>10}  {sign}{pct:>6.1f}%  {_mo(mos):>10}")
            print(f"{'═'*W}")

        if args.plot:
            plot_backtest(ticker, d, benchmarks, "all", args.days, "All Models")

    else:
        label = model_key.upper().replace("-", " ")
        _header(ticker, label, price)
        result = run_one(model_key, d, benchmarks)

        if args.plot:
            plot_backtest(ticker, d, benchmarks, model_key, args.days, label)

    print()


if __name__ == "__main__":
    main()
