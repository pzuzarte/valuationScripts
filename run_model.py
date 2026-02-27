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

Examples
--------
  python run_model.py NVDA dcf
  python run_model.py AAPL all
  python run_model.py MSFT monte-carlo --wacc 0.09
  python run_model.py GOOG peg --growth 0.18
"""

import sys, os, argparse, math
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
    else:
        label = model_key.upper().replace("-", " ")
        _header(ticker, label, price)
        run_one(model_key, d, benchmarks)

    print()


if __name__ == "__main__":
    main()
