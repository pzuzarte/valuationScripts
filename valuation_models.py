"""
valuation_models.py — Canonical valuation model library.

Contains all pure-calculation functions for stock price modeling:
  - WACC & cost of capital
  - Growth-adjusted multiples
  - Classic valuation: DCF, P/E, P/FCF, EV/EBITDA
  - Growth valuation: Reverse DCF, Forward PEG, EV/NTM Revenue,
                      TAM Scenario, Rule of 40, ERG, Graham Number
  - Convergence & reliability analysis
  - Value screening: trap flags, PEG, rank scoring
  - Growth screening: pillar scores, price targets, sentiment

All functions are pure calculations (no I/O, no API calls, no printing).
They accept a pre-populated data dict `d` and return results.

Canonical dict keys (as used in valuationMaster.py / fetch_tv_data):
  d["price"]          - current stock price
  d["market_cap"]     - market cap in dollars
  d["fcf"]            - trailing twelve-month free cash flow
  d["fcf_per_share"]  - FCF per share
  d["eps"]            - TTM diluted EPS
  d["fwd_eps"]        - forward (NTM) EPS
  d["ebitda"]         - TTM EBITDA
  d["shares"]         - shares outstanding
  d["total_debt"]     - total debt
  d["cash"]           - cash and equivalents
  d["est_growth"]     - estimated forward growth rate (decimal, e.g. 0.20)
  d["beta"]           - 1-year beta from TradingView
  d["wacc_raw"]       - dict of yfinance WACC inputs (beta_yf, interest_expense, etc.)
  d["wacc_override"]  - optional manual WACC override (decimal)
  d["revenue"]        - TTM total revenue
  d["gross_margin"]   - gross margin as a plain percentage (e.g. 65.0 for 65%)
  d["fcf_margin"]     - FCF margin as a plain percentage (e.g. 15.0 for 15%)
  d["net_margin"]     - net margin as a plain percentage (e.g. 20.0 for 20%)
  d["rev_growth"]     - TTM revenue growth rate (decimal)
  d["ntm_revenue"]    - next-twelve-months revenue estimate
  d["ntm_eps"]        - next-twelve-months EPS estimate
"""

import math

try:
    import numpy as np
    _NUMPY_AVAIL = True
except ImportError:
    np = None
    _NUMPY_AVAIL = False

# ── § 1  MODULE CONSTANTS ─────────────────────────────────────────────────────

PROJECTION_YEARS       = 5
TERMINAL_GROWTH_RATE   = 0.03
MARGIN_OF_SAFETY       = 0.20
RISK_FREE_RATE         = 0.043
EQUITY_RISK_PREMIUM    = 0.055
CONVERGENCE_THRESHOLD  = 0.15
DECAY_RATE             = 0.15
TERM_PE                = 25.0
TERM_PE_PROXY          = 20.0


# ── § 2  WACC ─────────────────────────────────────────────────────────────────

def calculate_wacc(d: dict) -> tuple:
    """
    Compute WACC using the textbook formula:
        WACC = (E/V) x Ke  +  (D/V) x Kd x (1 - t)

    Components:
        E   = market cap          (market-value equity weight)
        D   = total_debt          (from TradingView screener)
        Ke  = Rf + beta x ERP     (CAPM cost of equity)
        Kd  = interest_expense / total_debt   (from yfinance TTM)
              fallback: Rf + 1.5% investment-grade spread
        t   = income_tax / pretax_income      (from yfinance TTM)
              fallback: 21% US statutory rate

    Returns (wacc_rate: float, wacc_source: str)
    """
    # ── 0. Manual override wins unconditionally ──────────────────────────
    if d.get("wacc_override"):
        return round(d["wacc_override"], 4), "manual override"

    raw    = d.get("wacc_raw", {}) or {}
    mktcap = d.get("market_cap") or 0.0

    # Prefer 5-year monthly beta from yfinance (standard for WACC);
    # 1-year beta from TradingView is too noisy for discount rate purposes.
    beta_tv  = d.get("beta", 1.0) or 1.0
    beta_yf  = raw.get("beta_yf")
    beta     = beta_yf if (beta_yf is not None) else beta_tv
    beta_src = "yfinance 5Y" if (beta_yf is not None) else "TradingView 1Y"

    # Prefer yfinance balance sheet debt (more reliable than TV screener for WACC)
    debt_yf = raw.get("total_debt_yf")
    debt_tv = d.get("total_debt") or 0.0
    debt    = debt_yf if (debt_yf is not None and debt_yf > 0) else debt_tv
    debt_src = "yfinance BS" if (debt_yf is not None and debt_yf > 0) else "TradingView"

    int_exp  = raw.get("interest_expense")    # positive dollar value from yfinance
    tax_exp  = raw.get("income_tax_expense")  # may be negative (tax benefit)
    pretax   = raw.get("pretax_income")

    # ── 1. Cost of equity via CAPM ───────────────────────────────────────
    ke = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

    # ── 2. Effective tax rate ─────────────────────────────────────────────
    tax_rate   = 0.21
    tax_source = "21% statutory fallback"
    if pretax and pretax > 0 and tax_exp is not None:
        eff = max(0.0, tax_exp) / pretax      # floor at 0 for tax-benefit years
        if 0.0 <= eff <= 0.55:               # sanity bounds
            tax_rate   = eff
            tax_source = "yfinance TTM (tax / pretax_income)"

    # ── 3. Cost of debt ───────────────────────────────────────────────────
    kd        = RISK_FREE_RATE + 0.015        # fallback: Rf + 1.5% spread
    kd_source = "Rf + 1.5% spread (fallback)"
    if int_exp and int_exp > 0 and debt > 0:
        kd_raw = int_exp / debt
        if 0.01 <= kd_raw <= 0.20:           # sanity bounds: 1%–20%
            kd        = kd_raw
            kd_source = "yfinance TTM (interest_expense / total_debt)"

    # ── 4. Capital-structure weights ──────────────────────────────────────
    # Always run the full formula if we have market cap.
    # If debt is zero, wd=0 and the formula reduces to Ke naturally.
    if mktcap > 0:
        V  = mktcap + debt
        we = mktcap / V
        wd = debt   / V
        wacc = we * ke + wd * kd * (1.0 - tax_rate)
        if debt > 0:
            weight_source = "market-cap weighting (E={:.0f}%, D={:.0f}%)".format(
                we * 100, wd * 100)
        else:
            weight_source = "equity-only (TV total_debt=0 for this ticker)"
    else:
        wacc = ke
        weight_source = "CAPM only (no market cap data)"

    # ── 5. Sanity clamp: 6%–18% ───────────────────────────────────────────
    wacc = round(max(0.06, min(0.18, wacc)), 4)

    source = (
        "WACC: Ke={ke:.1f}% (CAPM β={beta:.2f} {bsrc}), Kd={kd:.1f}% ({kds}), "
        "t={t:.1f}% ({ts}), {ws}".format(
            ke=ke*100, beta=beta, bsrc=beta_src,
            kd=kd*100, kds=kd_source,
            t=tax_rate*100, ts=tax_source, ws=weight_source)
    )

    return wacc, source


# ── § 3  MULTIPLE SCALING ─────────────────────────────────────────────────────

def growth_adjusted_multiples(growth: float) -> dict:
    """
    Return target and conservative multiples scaled to the company's
    estimated growth rate, so fast-growing companies are not penalised
    by static market-average multiples.

    Anchors used:
      P/E:       Target = growth_pct * 1.5  (PEG of 1.5 = fair price for growth)
                 Conserv = growth_pct * 1.0  (PEG of 1.0 = no premium)
      P/FCF:     Target = growth_pct * 1.6  (slight FCF quality premium)
                 Conserv = growth_pct * 1.0
      EV/EBITDA: Target = 8 + growth_pct    (base multiple + 1x per growth point)
                 Conserv = 4 + growth_pct * 0.6

    All multiples are capped to reasonable min/max ranges.
    """
    g = growth * 100   # e.g. 0.20 -> 20.0

    return {
        "target_pe":    max(15.0, min(g * 1.5,  80.0)),
        "conserv_pe":   max(12.0, min(g * 1.0,  45.0)),
        "target_pfcf":  max(15.0, min(g * 1.6,  80.0)),
        "conserv_pfcf": max(12.0, min(g * 1.0,  45.0)),
        "target_eveb":  max(10.0, min(8 + g,     60.0)),
        "conserv_eveb": max(6.0,  min(4 + g * 0.6, 35.0)),
    }


# ── § 4  CLASSIC VALUATION METHODS ───────────────────────────────────────────

def run_dcf(d: dict) -> dict:
    """
    2-stage 10-year FCF DCF:
      Stage 1 (years 1-5): growth decays by 15% each year (compounding)
      Stage 2 (years 6-10): linear interpolation from stage-1 exit to terminal
      Terminal: Gordon growth model at TERMINAL_GROWTH_RATE

    Returns a rich dict with fair_value, mos_value, reliability flags, and details.
    Source: 10-year logic from portfolioAnalyzer.py; rich-dict return from valuationMaster.py.
    """
    fcf    = d.get("fcf")
    shares = d.get("shares")
    nd     = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    g      = d.get("est_growth") or 0.05

    if not (fcf and fcf > 0 and shares and shares > 0):
        return {
            "method": "DCF", "fair_value": None, "mos_value": None,
            "reliable": False, "debt_heavy": False, "warning": "Insufficient data",
        }

    wacc, wacc_source = calculate_wacc(d)

    # Stage 1: years 1-5, growth decays 15% per year (compounding)
    g1 = g
    cf = fcf
    pvs = []
    for yr in range(1, 6):
        g1 = g1 * (1 - DECAY_RATE)
        g1 = max(g1, TERMINAL_GROWTH_RATE * 2)
        cf = cf * (1 + g1)
        pvs.append(cf / (1 + wacc) ** yr)
    stage1_exit_growth = g1

    # Stage 2: years 6-10, linear decay to terminal growth
    for yr in range(6, 11):
        t  = (yr - 5) / 5
        g2 = stage1_exit_growth * (1 - t) + TERMINAL_GROWTH_RATE * t
        cf = cf * (1 + g2)
        pvs.append(cf / (1 + wacc) ** yr)

    sum_pv   = sum(pvs)
    term_val = cf * (1 + TERMINAL_GROWTH_RATE) / (wacc - TERMINAL_GROWTH_RATE)
    pv_term  = term_val / (1 + wacc) ** 10
    ev       = sum_pv + pv_term
    eq_val   = ev - nd

    debt_heavy = nd > ev * 0.8
    reliable   = eq_val > 0

    if eq_val <= 0:
        iv = d.get("price", 0.0)
        warning = "DEBT-HEAVY: Net debt exceeds DCF enterprise value. DCF unreliable — use EV/EBITDA instead."
    else:
        iv = round(eq_val / shares, 2)
        warning = "Debt load is significant — treat DCF with caution." if debt_heavy else None

    return {
        "method":      "DCF",
        "fair_value":  iv,
        "mos_value":   iv * (1 - MARGIN_OF_SAFETY),
        "wacc":        wacc,
        "wacc_source": wacc_source,
        "growth":      g,
        "reliable":    reliable,
        "debt_heavy":  debt_heavy,
        "warning":     warning,
        "net_debt":    nd,
        "details": {
            "sum_pv":   sum_pv,
            "pv_term":  pv_term,
            "ev":       ev,
            "eq_val":   max(eq_val, 0.0),
            "term_pct": (pv_term / ev * 100) if ev > 0 else 0,
        },
    }


def run_pfcf(d: dict, benchmarks: dict) -> dict:
    fcf_ps   = d["fcf_per_share"]
    mkt_mult = benchmarks["pfcf"]
    mults    = growth_adjusted_multiples(d["est_growth"])
    peer_label = benchmarks.get("sector_name", "Sector Median")

    fv_target  = fcf_ps * mults["target_pfcf"]
    fv_market  = fcf_ps * mkt_mult
    fv_conserv = fcf_ps * mults["conserv_pfcf"]
    midpoint   = (fv_target + fv_market + fv_conserv) / 3
    return {
        "method":           "P/FCF",
        "fair_value":       midpoint,
        "mos_value":        midpoint * (1 - MARGIN_OF_SAFETY),
        "fv_target":        fv_target,
        "fv_market":        fv_market,
        "fv_conserv":       fv_conserv,
        "mkt_multiple":     mkt_mult,
        "target_multiple":  mults["target_pfcf"],
        "conserv_multiple": mults["conserv_pfcf"],
        "peer_label":       peer_label,
    }


def run_pe(d: dict, benchmarks: dict) -> dict:
    eps     = d["eps"]
    mkt_pe  = benchmarks["pe"]
    mults   = growth_adjusted_multiples(d["est_growth"])
    peer_label = benchmarks.get("sector_name", "Sector Median")

    fv_target  = eps * mults["target_pe"]
    fv_market  = eps * mkt_pe
    fv_conserv = eps * mults["conserv_pe"]
    midpoint   = (fv_target + fv_market + fv_conserv) / 3
    return {
        "method":       "P/E",
        "fair_value":   midpoint,
        "mos_value":    midpoint * (1 - MARGIN_OF_SAFETY),
        "fv_target":    fv_target,
        "fv_market":    fv_market,
        "fv_conserv":   fv_conserv,
        "mkt_multiple": mkt_pe,
        "peg":          d.get("peg"),
        "target_mult":  mults["target_pe"],
        "conserv_mult": mults["conserv_pe"],
        "peer_label":   peer_label,
    }


def run_ev_ebitda(d: dict, benchmarks: dict) -> dict:
    ebitda   = d["ebitda"]
    nd       = d["total_debt"] - d["cash"]   # net debt
    shares   = d["shares"]
    mkt_mult = benchmarks["ev_ebitda"]
    mults    = growth_adjusted_multiples(d["est_growth"])
    peer_label = benchmarks.get("sector_name", "Sector Median")

    def ip(mult):
        return (ebitda * mult - nd) / shares

    fv_target  = ip(mults["target_eveb"])
    fv_market  = ip(mkt_mult)
    fv_conserv = ip(mults["conserv_eveb"])
    midpoint   = (fv_target + fv_market + fv_conserv) / 3
    return {
        "method":        "EV/EBITDA",
        "fair_value":    midpoint,
        "mos_value":     midpoint * (1 - MARGIN_OF_SAFETY),
        "fv_target":     fv_target,
        "fv_market":     fv_market,
        "fv_conserv":    fv_conserv,
        "mkt_multiple":  mkt_mult,
        "target_mult":   mults["target_eveb"],
        "conserv_mult":  mults["conserv_eveb"],
        "ebitda":        ebitda,
        "ebitda_method": d["ebitda_method"],
        "peer_label":    peer_label,
    }


# ── § 5  GROWTH VALUATION METHODS ────────────────────────────────────────────

def calibrate_erg_multiple(peer_data: dict, d: dict) -> tuple:
    """
    Position the subject company within the peer ERG distribution using two
    quality dimensions:

        1. Gross margin percentile vs. peers
           Higher-margin businesses earn more EV per unit of growth because
           each marginal revenue dollar drops through to the bottom line more
           efficiently.

        2. Rule-of-40 percentile vs. peers
           Captures combined growth + profitability quality — the standard
           institutional benchmark for tech/software companies.

    The two signals are blended 50/50 into a composite quality percentile,
    then that percentile is looked up in the peer ERG distribution.
    Result is clipped to [10th, 90th] percentile of peers so we never assign
    a truly extreme outlier multiple.

    Returns (calibrated_erg: float, details: dict).
    """
    peers      = peer_data.get("multiples", [])
    median_erg = peer_data.get("median", 0.50)

    if not peers:
        return median_erg, {
            "calibrated_erg": median_erg,
            "gm_pctile": 50.0, "ro40_pctile": 50.0,
            "composite_pctile": 50.0,
            "note": "No peer data — median fallback used",
        }

    # Subject quality metrics
    subj_gm   = d.get("gross_margin") or 50.0
    # FCF margin from data dict
    fcf       = d.get("fcf") or 0.0
    rev       = d.get("revenue") or 1.0
    rev_g     = (d.get("rev_growth_pct") or d.get("est_growth", 0.10) * 100)
    fcf_m_pct = (fcf / rev * 100) if (fcf and rev > 0) else 0.0
    subj_ro40 = rev_g + fcf_m_pct

    # Peer distribution arrays
    peer_gm_vals   = sorted(r["gm"]   for r in peers if r["gm"]   > 0)
    peer_ro40_vals = sorted(r["ro40"] for r in peers)
    peer_erg_vals  = sorted(r["erg"]  for r in peers)

    def _rank(val, arr):
        """Percentile rank of val in a sorted list (0–100)."""
        if not arr: return 50.0
        return sum(1 for x in arr if x < val) / len(arr) * 100.0

    def _at_pctile(pct, arr):
        """Value at a given percentile in a sorted list."""
        if not arr: return median_erg
        idx = (pct / 100.0) * (len(arr) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(arr) - 1)
        return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)

    gm_pctile   = _rank(subj_gm,   peer_gm_vals)   if peer_gm_vals   else 50.0
    ro40_pctile = _rank(subj_ro40, peer_ro40_vals) if peer_ro40_vals else 50.0

    # 50/50 blend, clipped to [10, 90] — never assign the most extreme peer multiple
    composite_pctile = max(10.0, min(90.0, gm_pctile * 0.5 + ro40_pctile * 0.5))

    calibrated_erg = _at_pctile(composite_pctile, peer_erg_vals)
    # Absolute sanity bounds
    calibrated_erg = max(0.03, min(calibrated_erg, 3.0))

    return calibrated_erg, {
        "calibrated_erg":    round(calibrated_erg, 4),
        "gm_pctile":         round(gm_pctile, 1),
        "ro40_pctile":       round(ro40_pctile, 1),
        "composite_pctile":  round(composite_pctile, 1),
        "subj_gm":           round(subj_gm, 1),
        "subj_ro40":         round(subj_ro40, 1),
        "peer_erg_p25":      peer_data.get("p25", 0.30),
        "peer_erg_median":   peer_data.get("median", 0.50),
        "peer_erg_p75":      peer_data.get("p75", 0.70),
        "peer_count":        peer_data.get("peer_count", 0),
        "sector_name":       peer_data.get("sector_name", ""),
        "fallback":          peer_data.get("fallback", True),
        "note": (
            "GM {gm:.1f}% → {gmp:.0f}th pctile  |  "
            "Ro40 {ro40:.1f} → {r40p:.0f}th pctile  |  "
            "Composite → {cp:.0f}th pctile  |  "
            "ERG dist: {p25:.3f} / {med:.3f} / {p75:.3f} (p25/med/p75)"
        ).format(
            gm=subj_gm, gmp=gm_pctile,
            ro40=subj_ro40, r40p=ro40_pctile,
            cp=composite_pctile,
            p25=peer_data.get("p25", 0.30),
            med=peer_data.get("median", 0.50),
            p75=peer_data.get("p75", 0.70),
        ),
    }


def run_reverse_dcf(d):
    # NOTE: Uses a simpler 5-year constant-growth DCF (not the 10-year two-stage
    # decay model in run_dcf). The implied growth rate returned here is therefore
    # NOT directly comparable as an input to run_dcf.
    price=d["price"]; fcf=d["fcf"]; shares=d["shares"]
    if not(price and fcf and fcf>0 and shares and shares>0): return None
    nd=d["total_debt"]-d["cash"]
    wacc, _ = calculate_wacc(d)
    def dcf_at(g):
        cf,pvs=fcf,[]
        for yr in range(1,PROJECTION_YEARS+1):
            cf*=(1+g); pvs.append(cf/(1+wacc)**yr)
        last=fcf*(1+g)**PROJECTION_YEARS
        pv_tv=(last*(1+TERMINAL_GROWTH_RATE)/(wacc-TERMINAL_GROWTH_RATE))/(1+wacc)**PROJECTION_YEARS
        eq=sum(pvs)+pv_tv-nd; return(eq/shares) if eq>0 else 0.0
    lo,hi=-0.10,1.50
    for _ in range(60):
        mid=(lo+hi)/2
        if dcf_at(mid)<price: lo=mid
        else: hi=mid
    ig=round((lo+hi)/2,4); sg=d["est_growth"]
    scen=[{"label":"Bear","g":max(0.0,ig-0.10)},{"label":"Implied","g":ig},
          {"label":"Street","g":sg},{"label":"Bull","g":min(1.50,ig+0.15)}]
    for s in scen: s["fv"]=dcf_at(s["g"]); s["upside"]=((s["fv"]-price)/price*100) if price else 0
    return{"method":"Reverse DCF","implied_growth":ig,"street_growth":sg,"wacc":wacc,"scenarios":scen}


def run_forward_peg(d: dict) -> dict:
    """
    Forward PEG valuation — prices NTM earnings at a PEG ratio of 1.0.

    Three scenarios based on PEG multiple applied to NTM EPS × growth%:
      Conservative: PEG 1.0x  (base × 1.0)
      Base:         PEG 1.5x  (base × 1.5)
      Premium:      PEG 2.0x  (base × 2.0)

    Returns a dict with fair_value set to the base (PEG 1.5×) scenario,
    plus fv_conserv, fv_base, fv_premium for the full range.
    """
    eps      = d["eps"]
    fwd_eps  = d.get("fwd_eps")
    growth   = d["est_growth"]
    price    = d["price"]

    if fwd_eps and fwd_eps > 0:
        ntm_eps    = fwd_eps
        eps_source = "analyst NTM consensus"
    elif eps and eps > 0:
        ntm_eps    = eps * (1 + growth)
        eps_source = "TTM EPS grown " + format(growth * 100, ".1f") + "%"
    else:
        return None

    g_pct = growth * 100
    if g_pct <= 0:
        return None

    cap   = price * 5 if price else 9999
    fv_c  = max(0, min(ntm_eps * g_pct * 1.0, cap))
    fv_b  = max(0, min(ntm_eps * g_pct * 1.5, cap))
    fv_p  = max(0, min(ntm_eps * g_pct * 2.0, cap))
    cpeg  = (price / ntm_eps / g_pct) if (ntm_eps > 0 and price) else None

    return {
        "method":          "Fwd PEG",
        "fair_value":      fv_b,
        "mos_value":       fv_b * (1 - MARGIN_OF_SAFETY),
        "ntm_eps":         ntm_eps,
        "eps_source":      eps_source,
        "growth":          growth,
        "fv_conserv":      fv_c,
        "fv_base":         fv_b,
        "fv_premium":      fv_p,
        "current_fwd_peg": cpeg,
    }


def run_ev_ntm_revenue(d):
    rev        = d["revenue"]
    fwd_rev    = d.get("fwd_rev")
    growth     = d["est_growth"]
    gm         = d["gross_margin"]
    nd         = d["total_debt"] - d["cash"]
    shares     = d["shares"]
    price      = d["price"]

    if not (rev and rev > 0 and shares and shares > 0):
        return None

    ntm_rev    = fwd_rev if (fwd_rev and fwd_rev > 0) else rev * (1 + growth)
    rev_source = (
        "analyst NTM consensus"
        if (fwd_rev and fwd_rev > 0)
        else "TTM grown " + format(growth * 100, ".1f") + "%"
    )

    # ── Step 1: Gross-margin tier base ranges (wider than before) ──
    if gm and gm >= 70:
        base_lo, base_mid, base_hi = 8.0, 15.0, 25.0
        tier = "High-margin (GM ≥70%)"
    elif gm and gm >= 50:
        base_lo, base_mid, base_hi = 4.0, 8.0, 16.0
        tier = "Mid-high margin (GM 50–70%)"
    elif gm and gm >= 30:
        base_lo, base_mid, base_hi = 2.5, 4.5, 9.0
        tier = "Mid margin (GM 30–50%)"
    else:
        base_lo, base_mid, base_hi = 1.0, 2.5, 5.0
        tier = "Low margin (GM <30%)"

    # ── Step 2: Growth bonus — scales aggressively for hypergrowth ──
    # +1x mid per 10ppt growth above 15%, accelerating above 40%
    g_pct = growth * 100
    if g_pct > 15:
        linear    = (g_pct - 15) / 10 * 1.5           # +1.5x per 10ppt above 15%
        accel     = max(0, (g_pct - 40) / 10) ** 1.3   # accelerating bonus above 40%
        bonus_mid = linear + accel
        bonus_lo  = bonus_mid * 0.5
        bonus_hi  = bonus_mid * 1.4
    else:
        bonus_mid = 0.0
        bonus_lo  = 0.0
        bonus_hi  = 0.0

    tier_lo  = base_lo  + bonus_lo
    tier_mid = base_mid + bonus_mid
    tier_hi  = base_hi  + bonus_hi

    # ── Step 3: Anchor against current market EV/Revenue ──
    # The market's current multiple is information — if it's pricing the stock
    # at 28x revenue, our model shouldn't cap at 20x.  We blend the tier-derived
    # range with the market multiple so the output brackets reality.
    current_ev_rev = (d["ev_approx"] / rev) if rev > 0 else None
    anchor_source  = "tier-only"

    if current_ev_rev and current_ev_rev > 1.0:
        mkt = current_ev_rev
        # Blend: 60% tier-derived, 40% market-anchored
        # This respects our fundamental view while staying grounded
        lo  = tier_lo  * 0.6 + mkt * 0.70 * 0.4   # bear = 70% of market
        mid = tier_mid * 0.6 + mkt * 1.00 * 0.4   # base = market level
        hi  = tier_hi  * 0.6 + mkt * 1.25 * 0.4   # bull = 125% of market
        anchor_source = "blended (60% tier + 40% market)"
    else:
        lo, mid, hi = tier_lo, tier_mid, tier_hi

    # ── Step 4: Apply reasonable floor/ceiling per tier ──
    lo  = max(1.0, round(lo,  1))
    mid = max(2.0, round(mid, 1))
    hi  = max(3.0, round(hi,  1))

    def fv_at(m):
        eq = ntm_rev * m - nd
        return (eq / shares) if eq > 0 else 0.0

    return {
        "method":          "EV/NTM Rev",
        "fair_value":      fv_at(mid),
        "mos_value":       fv_at(mid) * (1 - MARGIN_OF_SAFETY),
        "ntm_rev":         ntm_rev,
        "rev_source":      rev_source,
        "tier":            tier,
        "gross_margin":    gm,
        "mult_lo":         round(lo,  1),
        "mult_mid":        round(mid, 1),
        "mult_hi":         round(hi,  1),
        "growth_bonus":    round(bonus_mid, 2),
        "anchor_source":   anchor_source,
        "fv_lo":           fv_at(lo),
        "fv_mid":          fv_at(mid),
        "fv_hi":           fv_at(hi),
        "current_ev_rev":  current_ev_rev,
    }


def run_tam_scenario(d):
    rev=d["revenue"]; growth=d["est_growth"]; gm=d["gross_margin"] or 50.0
    nd=d["total_debt"]-d["cash"]; shares=d["shares"]; price=d["price"]
    if not(rev and rev>0 and shares and shares>0): return None
    wacc, _ = calculate_wacc(d)
    tam_m=min(10+max(0,(growth-0.20)/0.10*2.5),20.0); tam=rev*tam_m

    # ── Net margin: use actual TTM net margin where available ──────────
    # Deriving from gross margin (gm*0.35) systematically underestimates
    # high-margin companies like NVDA (actual ~55% NM vs 17.5% from formula).
    # Priority: (1) actual net income / revenue, (2) gross-margin proxy.
    ni = d.get("net_income")
    if ni and rev and rev > 0 and ni / rev > 0.03:
        # Use actual net margin, clipped to a reasonable range [5%, 60%]
        actual_nm = max(0.05, min(0.60, ni / rev))
        base_nm = actual_nm
        nm_source = "actual TTM"
    else:
        # Fallback: scale up the GM→NM ratio for high-margin businesses.
        # For GM ≥ 70%, use 0.45× (captures high-operating-leverage cos).
        # For GM ≥ 50%, use 0.38×. Below 50%, use 0.30× (asset-heavy).
        gm_val = gm / 100.0
        if gm_val >= 0.70:
            ratio = 0.45
        elif gm_val >= 0.50:
            ratio = 0.38
        else:
            ratio = 0.30
        base_nm = max(0.05, gm_val * ratio)
        nm_source = "GM proxy"

    # ── Market share scenarios: anchor to company's implied trajectory ─
    # Classic 4%/8%/15% caps work for early-stage companies targeting a TAM.
    # For larger/faster-growing companies those caps may already be below
    # the company's current or near-term market share, making the model useless.
    # Instead, compute the implied year-5 market share from current growth rate
    # and set bear/base/bull around that anchor.
    yr5_implied_rev = rev * (1 + max(growth, 0.0)) ** 5
    implied_share = yr5_implied_rev / tam if tam > 0 else 0.08

    # Cap implied share at 65% — even dominant monopolies rarely exceed this.
    implied_share = min(0.65, max(0.04, implied_share))

    # Bear = 60% of base, Bull = 150% of base, floored/capped sensibly.
    base_share = round(max(0.04, min(0.55, implied_share)),      3)
    bear_share = round(max(0.02, min(0.40, implied_share * 0.6)), 3)
    bull_share = round(max(0.08, min(0.65, implied_share * 1.5)), 3)

    scen=[]
    for label, share_cap, mm in [("Bear", bear_share, 0.70),
                                   ("Base", base_share, 1.00),
                                   ("Bull", bull_share, 1.30)]:
        yr5e=tam*share_cap*base_nm*mm; pv=yr5e*TERM_PE/(1+wacc)**5
        # yr5e is net income (post-interest), so P/E × net income = equity value.
        # No debt deduction needed — net income already reflects interest expense.
        fv=max(0.0,pv/shares)
        scen.append({"label":label,"share_pct":share_cap*100,"yr5_rev":tam*share_cap,
                     "net_margin":base_nm*mm*100,"yr5_earn":yr5e,"fv":fv,
                     "upside":((fv-price)/price*100) if price else 0})
    bf=scen[1]["fv"]
    return{"method":"TAM Scenario","fair_value":bf,"mos_value":bf*(1-MARGIN_OF_SAFETY),
           "tam_est":tam,"tam_mult":round(tam_m,1),"base_net_margin":base_nm*100,
           "nm_source":nm_source,"implied_share_pct":round(implied_share*100,1),
           "wacc":wacc,"scenarios":scen}


def run_rule_of_40(d):
    rev=d["revenue"]; fwd_rev=d.get("fwd_rev"); growth=d["est_growth"]
    fcf_mar=d["fcf_margin"]; nd=d["total_debt"]-d["cash"]; shares=d["shares"]; price=d["price"]
    rev_g=d.get("rev_growth_pct")
    if not(rev and rev>0 and shares and shares>0): return None
    rg=rev_g if rev_g is not None else growth*100
    fm=(fcf_mar*100) if fcf_mar is not None else 0.0; ro40=rg+fm
    if ro40>=60:   cohort,lo,mid,hi="Elite (≥60)",       15.0,20.0,25.0
    elif ro40>=40: cohort,lo,mid,hi="Strong (40–60)",     8.0,11.0,15.0
    elif ro40>=20: cohort,lo,mid,hi="Average (20–40)",    4.0, 6.0, 8.0
    else:          cohort,lo,mid,hi="Below average (<20)",1.0, 2.5, 4.0
    ntm=fwd_rev if(fwd_rev and fwd_rev>0) else rev*(1+growth)
    def fv_at(m): eq=ntm*m-nd; return(eq/shares) if eq>0 else 0.0
    return{"method":"Rule of 40","fair_value":fv_at(mid),"mos_value":fv_at(mid)*(1-MARGIN_OF_SAFETY),
           "ro40":round(ro40,1),"rev_growth_pct":round(rg,1),"fcf_margin_pct":round(fm,1),
           "cohort":cohort,"mult_lo":lo,"mult_mid":mid,"mult_hi":hi,
           "fv_lo":fv_at(lo),"fv_mid":fv_at(mid),"fv_hi":fv_at(hi),"ntm_rev":ntm,
           "current_ev_rev":(d["ev_approx"]/rev) if rev>0 else None}


def run_erg_valuation(d, erg_peer_data=None):
    """
    ERG — dual-approach Growth Valuation.

    Combines two complementary lenses into a single blended fair value:

    ── Approach A: Earnings Build (DCF-style) ──────────────────────
    Projects 5-year revenue at the estimated growth rate, applies a maturity
    net-margin target, multiplies by a growth-tier terminal P/E, and discounts
    back to present value.  Designed for pre-peak-margin companies where current
    earnings understate long-run earning power.

    ── Approach B: EV/Revenue-to-Growth (market-calibrated) ────────
    Uses a live, peer-derived ERG multiple (EV/Revenue ÷ growth%) to value
    the NTM revenue stream.  The multiple is NOT hardcoded — it is computed
    from actual sector peers in TradingView and then quality-adjusted to
    reflect where this company sits within its peer distribution on gross
    margin and Rule-of-40 score.  This is analogous to using a median P/E
    rather than a fixed P/E.

        peer_erg_i      = (EV_i / Revenue_i) / rev_growth_pct_i
        calibrated_erg  = percentile(peer_erg, composite_quality_rank)
        Fair EV (B)     = calibrated_erg × growth% × NTM Revenue

    ── Blending ────────────────────────────────────────────────────
    Base fair value = 60% Approach A (intrinsic) + 40% Approach B (market).
    This respects fundamental value while anchoring to what the market is
    currently paying for comparable growth in the same sector.

    Returns None if revenue or shares are unavailable.
    """
    rev     = d["revenue"]
    growth  = d["est_growth"]
    gm      = d.get("gross_margin") or 50.0
    ni      = d.get("net_income")
    nd      = d["total_debt"] - d["cash"]
    shares  = d["shares"]
    price   = d["price"]
    fwd_rev = d.get("fwd_rev")
    if not (rev and rev > 0 and shares and shares > 0):
        return None
    wacc, _ = calculate_wacc(d)

    g_pct   = growth * 100
    ntm_rev = fwd_rev if (fwd_rev and fwd_rev > 0) else rev * (1 + growth)

    # ══════════════════════════════════════════════════════════════
    #  APPROACH A — Earnings Build
    # ══════════════════════════════════════════════════════════════

    # Terminal P/E tier
    if g_pct >= 30:
        te_bull, te_base, te_bear = 28.0, 23.0, 18.0
        tier_label = "High-growth (≥30%)"
    elif g_pct >= 15:
        te_bull, te_base, te_bear = 23.0, 18.0, 15.0
        tier_label = "Mid-growth (15–30%)"
    else:
        te_bull, te_base, te_bear = 18.0, 15.0, 12.0
        tier_label = "Low-growth (<15%)"

    # Mature net margin target
    if ni and rev > 0 and ni / rev > 0.05:
        ttm_nm    = max(0.05, min(0.60, ni / rev))
        expansion = min(0.12, g_pct / 100 * 0.3)
        nm_base   = min(0.60, ttm_nm + expansion)
        nm_bear   = max(0.05, ttm_nm * 0.85)
        nm_bull   = min(0.65, ttm_nm + expansion * 1.6)
        nm_source = "TTM NM {:.1f}% + expansion".format(ttm_nm * 100)
    else:
        gm_frac = gm / 100.0
        if gm_frac >= 0.70:   base_ratio = 0.42
        elif gm_frac >= 0.50: base_ratio = 0.35
        else:                  base_ratio = 0.25
        nm_base   = max(0.05, min(0.55, gm_frac * base_ratio * (1 + g_pct / 200)))
        nm_bear   = nm_base * 0.75
        nm_bull   = min(0.60, nm_base * 1.30)
        nm_source = "GM {:.0f}% proxy".format(gm)

    # Project scenarios for Approach A
    scenarios_a = []
    for label, g_rate, nm, term_pe in [
        ("Bear", max(0.02, growth * 0.60), nm_bear, te_bear),
        ("Base", growth,                    nm_base, te_base),
        ("Bull", min(1.20, growth * 1.40), nm_bull, te_bull),  # +40% above base, symmetric with bear
    ]:
        yr5_rev     = rev * (1 + g_rate) ** 5
        yr5_earn    = yr5_rev * nm
        term_equity = yr5_earn * term_pe
        pv_equity   = term_equity / (1 + wacc) ** 5
        # yr5_earn is net income (post-interest), so term_pe × net income = equity value.
        # No debt deduction needed — net income already reflects interest expense.
        fv          = max(0.0, pv_equity / shares)
        upside      = (fv - price) / price * 100 if price else 0
        scenarios_a.append({
            "label": label, "g_rate": g_rate, "nm": nm, "term_pe": term_pe,
            "yr5_rev": yr5_rev, "yr5_earn": yr5_earn, "fv": fv, "upside": upside,
        })

    fv_a_bear = scenarios_a[0]["fv"]
    fv_a_base = scenarios_a[1]["fv"]
    fv_a_bull = scenarios_a[2]["fv"]

    # ══════════════════════════════════════════════════════════════
    #  APPROACH B — EV/Revenue-to-Growth (market-calibrated)
    # ══════════════════════════════════════════════════════════════

    erg_calib_details = None
    calibrated_erg    = None
    fv_b_bear = fv_b_base = fv_b_bull = None

    if erg_peer_data and erg_peer_data.get("peer_count", 0) >= 3:
        calibrated_erg, erg_calib_details = calibrate_erg_multiple(erg_peer_data, d)

        def _ev_rev_fv(erg_mult):
            ev  = erg_mult * g_pct * ntm_rev
            eq  = ev - nd
            return max(0.0, eq / shares) if shares > 0 else 0.0

        fv_b_bear = _ev_rev_fv(erg_peer_data.get("p25", calibrated_erg * 0.65))
        fv_b_base = _ev_rev_fv(calibrated_erg)
        fv_b_bull = _ev_rev_fv(erg_peer_data.get("p75", calibrated_erg * 1.40))

    # ══════════════════════════════════════════════════════════════
    #  BLEND: 60% Approach A + 40% Approach B (if B available)
    # ══════════════════════════════════════════════════════════════

    if fv_b_base is not None and fv_b_base > 0:
        fv_bear  = fv_a_bear * 0.60 + fv_b_bear * 0.40
        fv_base  = fv_a_base * 0.60 + fv_b_base * 0.40
        fv_bull  = fv_a_bull * 0.60 + fv_b_bull * 0.40
        blend_note = "60% Earnings Build + 40% EV/Rev-to-Growth (calibrated)"
    else:
        fv_bear  = fv_a_bear
        fv_base  = fv_a_base
        fv_bull  = fv_a_bull
        blend_note = "Earnings Build only (no peer ERG data available)"

    return {
        "method":            "ERG",
        "fair_value":        fv_base,
        "mos_value":         fv_base * (1 - MARGIN_OF_SAFETY),
        "wacc":              wacc,
        "growth":            growth,
        "nm_source":         nm_source,
        "tier_label":        tier_label,
        "nm_base":           nm_base,
        "nm_bear":           nm_bear,
        "nm_bull":           nm_bull,
        "te_base":           te_base,
        "te_bear":           te_bear,
        "te_bull":           te_bull,
        # Approach A outputs
        "fv_a_bear":         fv_a_bear,
        "fv_a_base":         fv_a_base,
        "fv_a_bull":         fv_a_bull,
        "scenarios_a":       scenarios_a,
        # Approach B outputs
        "calibrated_erg":    calibrated_erg,
        "erg_calib_details": erg_calib_details,
        "fv_b_bear":         fv_b_bear,
        "fv_b_base":         fv_b_base,
        "fv_b_bull":         fv_b_bull,
        "ntm_rev":           ntm_rev,
        # Blended outputs
        "fv_bear":           fv_bear,
        "fv_base":           fv_base,
        "fv_bull":           fv_bull,
        "blend_note":        blend_note,
        # Legacy aliases (backtest + other code uses these)
        "fv_bear_compat":    fv_bear,
        "fv_base_compat":    fv_base,
        "fv_bull_compat":    fv_bull,
    }


def run_graham_number(d: dict) -> float:
    """
    Benjamin Graham intrinsic value formula:
      FV = sqrt(22.5 × NTM_EPS × Book_Value_per_Share)

    Most relevant for value stocks and dividend-payers.
    Not appropriate for high-growth tech (will massively undervalue).
    Returns None for negative book value or growth stocks with no book.
    """
    fwd_eps = d.get("fwd_eps") or d.get("eps")
    if not (fwd_eps and fwd_eps > 0):
        return None
    shares = d.get("shares") or 0
    mktcap = d.get("market_cap") or 0
    total_debt = d.get("total_debt") or 0
    # Estimate book value from market cap proxy (rough — yfinance book value)
    # We use the wacc_raw field if equity was fetched
    book_ps = d.get("wacc_raw", {}).get("book_value_ps")
    if not (book_ps and book_ps > 0):
        return None
    if fwd_eps <= 0 or book_ps <= 0:
        return None
    fv = (22.5 * fwd_eps * book_ps) ** 0.5
    return round(fv, 2) if fv > 0 else None


# ── § 6  CONVERGENCE & RELIABILITY ───────────────────────────────────────────

def analyse_convergence(results: list, price: float) -> dict:
    fvs        = [r["fair_value"] for r in results]
    sorted_fvs = sorted(fvs)
    n          = len(sorted_fvs)
    median     = (sorted_fvs[n // 2] if n % 2 == 1
                  else (sorted_fvs[n // 2 - 1] + sorted_fvs[n // 2]) / 2)
    mean       = sum(fvs) / len(fvs)
    low    = min(fvs)
    high   = max(fvs)
    spread = (high - low) / mean if mean else 0

    # Which methods converge (within threshold of median)?
    converging = [r for r in results if abs(r["fair_value"] - median) / median <= CONVERGENCE_THRESHOLD]
    diverging  = [r for r in results if abs(r["fair_value"] - median) / median >  CONVERGENCE_THRESHOLD]

    # Upside/downside
    upside = ((mean - price) / price * 100) if price else 0

    # Conviction level
    if len(converging) == 4:
        conviction = "HIGH"
        conviction_color = "#00c896"
    elif len(converging) == 3:
        conviction = "MODERATE"
        conviction_color = "#f0a500"
    else:
        conviction = "LOW"
        conviction_color = "#e05c5c"

    # Overall verdict
    if   mean > price * 1.20: verdict = "UNDERVALUED"
    elif mean < price * 0.80: verdict = "OVERVALUED"
    else:                      verdict = "FAIRLY VALUED"

    return {
        "fvs":              fvs,
        "median":           median,
        "mean":             mean,
        "low":              low,
        "high":             high,
        "spread_pct":       spread * 100,
        "converging":       converging,
        "diverging":        diverging,
        "upside":           upside,
        "conviction":       conviction,
        "conviction_color": conviction_color,
        "verdict":          verdict,
    }


def assess_reliability(d: dict, classic_results: list, growth_results: dict) -> dict:
    """
    Evaluate each valuation method's reliability for *this specific stock*.

    Returns a dict keyed by method name, each containing:
      - "reliable":  True / False
      - "flags":     list of short warning strings (empty if reliable)
      - "reason":    one-sentence human-readable summary (None if reliable)

    A method is flagged as unreliable if ANY flag is raised.
    """
    price = d["price"] or 0
    flags = {}

    # ── Classic methods ────────────────────────────────────────────
    for r in classic_results:
        m = r["method"]
        f = []

        if m == "DCF":
            tp = r["details"].get("term_pct", 0)
            if tp > 85:
                f.append("Terminal value is {:.0f}% of EV — result is dominated by far-future assumptions".format(tp))
            if r.get("debt_heavy"):
                f.append("Net debt exceeds 80% of DCF enterprise value — equity residual is unreliable")
            if not r.get("reliable"):
                f.append("DCF produced negative equity value — model is not meaningful for this capital structure")
            g = r.get("growth", 0)
            if g > 0.50:
                f.append("Growth rate of {:.0f}% is very high — DCF is extremely sensitive to this assumption".format(g * 100))

        elif m == "P/E":
            pe = d.get("current_pe")
            eps = d.get("eps")
            fwd = d.get("fwd_eps")
            if pe and (pe > 80 or pe < 3):
                f.append("Current P/E of {:.1f}x is extreme — earnings may be distorted by one-time items".format(pe))
            if eps and fwd and eps > 0:
                chg = abs(fwd - eps) / eps
                if chg > 0.60:
                    f.append("Forward EPS differs from TTM by {:.0f}% — earnings are in rapid transition".format(chg * 100))
            ni = d.get("net_income")
            fcf = d.get("fcf")
            if ni and fcf and ni > 0 and abs(fcf - ni) / ni > 1.5:
                f.append("Large gap between net income and FCF — earnings quality may be poor")

        elif m == "P/FCF":
            fcf_m = d.get("fcf_margin")
            if fcf_m is not None and fcf_m < 0.03:
                f.append("FCF margin is only {:.1f}% — thin cash flows amplify valuation noise".format(fcf_m * 100))
            ni = d.get("net_income")
            fcf = d.get("fcf")
            if ni and fcf and fcf > 0 and ni > 0:
                ratio = fcf / ni
                if ratio > 3.0 or ratio < 0.25:
                    f.append("FCF/Net income ratio of {:.1f}x suggests one-time items distorting cash flows".format(ratio))

        elif m == "EV/EBITDA":
            em = d.get("ebitda_method", "")
            if "net income" in em.lower():
                f.append("EBITDA was estimated from net income × 1.35 — low confidence in the input")
            nd = d.get("total_debt", 0) - d.get("cash", 0)
            ev = d.get("ev_approx", 0)
            if ev > 0 and nd > ev * 0.6:
                f.append("Net debt is {:.0f}% of EV — high leverage makes equity residual volatile".format(nd / ev * 100))

        flags[m] = f

    # ── Growth methods ─────────────────────────────────────────────
    rdcf = growth_results.get("reverse_dcf")
    if rdcf:
        f = []
        ig = rdcf.get("implied_growth", 0)
        if ig > 1.0:
            f.append("Implied growth of {:.0f}% is unrealistically high — price may be driven by speculation".format(ig * 100))
        if ig < -0.05:
            f.append("Implied growth is negative — market expects FCF decline, making this less informative")
        flags["Reverse DCF"] = f

    fpeg = growth_results.get("forward_peg")
    if fpeg:
        f = []
        g = fpeg.get("growth", 0)
        if g < 0.05:
            f.append("Growth rate of {:.1f}% is very low — PEG ratio becomes unstable below ~5%".format(g * 100))
        src = fpeg.get("eps_source", "")
        if "grown" in src.lower():
            f.append("NTM EPS is projected (no analyst consensus) — lower confidence in the earnings anchor")
        if fpeg.get("fair_value", 0) > price * 5:
            f.append("Fair value is >5× current price — PEG may be overstating for this growth profile")
        flags["Fwd PEG"] = f

    evntm = growth_results.get("ev_ntm_revenue")
    if evntm:
        f = []
        fv = evntm.get("fair_value", 0)
        if price > 0 and fv > 0:
            gap = abs(fv - price) / price
            if gap > 0.70:
                f.append("Fair value is {:.0f}% away from current price — model inputs may not suit this stock".format(gap * 100))
        # Revenue multiples are less meaningful for highly profitable mature companies
        ni = d.get("net_income")
        rev = d.get("revenue")
        if ni and rev and rev > 0:
            net_margin = ni / rev
            growth = d.get("est_growth", 0)
            if net_margin > 0.25 and growth < 0.15:
                f.append("Net margin of {:.0f}% with low growth — this mature, profitable company is better valued on earnings".format(net_margin * 100))
        if evntm.get("anchor_source") == "tier-only":
            f.append("No market EV/Revenue anchor available — multiples are tier-estimated only")
        flags["EV/NTM Rev"] = f

    tam = growth_results.get("tam_scenario")
    if tam:
        f = []
        scen = tam.get("scenarios", [])
        if len(scen) >= 3:
            bear_fv = scen[0].get("fv", 0)
            bull_fv = scen[2].get("fv", 0)
            if bear_fv > 0 and bull_fv > 0:
                spread = (bull_fv - bear_fv) / bear_fv
                # TAM scenarios structurally span ~7× between bear/bull due to market-share
                # assumptions (4%→15%) and margin multipliers (0.70→1.30). Only flag truly
                # extreme spreads that suggest inputs are unreliable (threshold: 10×).
                if spread > 10.0:
                    f.append("Bull/bear spread is {:.0f}× — scenario range is extremely wide; treat all estimates as order-of-magnitude only".format(spread))
        tm = tam.get("tam_mult", 0)
        # tam_mult is capped at 20.0; only flag when it hits that cap (truly speculative)
        if tm >= 19.5:
            f.append("TAM is capped at 20× current revenue — company is very early-stage; estimates carry high uncertainty")
        flags["TAM Scenario"] = f

    ro = growth_results.get("rule_of_40")
    if ro:
        f = []
        gm = d.get("gross_margin")
        if gm and gm < 40:
            f.append("Gross margin of {:.0f}% suggests this isn't a SaaS/software business — Rule of 40 is less applicable".format(gm))
        ro40 = ro.get("ro40", 0)
        rev_g = ro.get("rev_growth_pct", 0)
        fcf_m = ro.get("fcf_margin_pct", 0)
        if rev_g < 5 and fcf_m < 5:
            f.append("Both revenue growth and FCF margin are very low — Rule of 40 lacks discriminating power here")
        flags["Rule of 40"] = f

    erg = growth_results.get("erg")
    if erg:
        f = []
        rev = d.get("revenue")
        if not rev or rev <= 0:
            f.append("No revenue data — ERG cannot project forward earnings")
        g = erg.get("growth", 0)
        if g < 0.05:
            f.append("Growth rate of {:.1f}% is very low — ERG's margin-expansion logic is unreliable below ~5%".format(g * 100))
        if g > 0.80:
            f.append("Growth rate of {:.0f}% is extremely high — ERG projections carry very wide uncertainty at this rate".format(g * 100))
        bear_fv = erg.get("fv_bear", 0)
        bull_fv = erg.get("fv_bull", 0)
        if bear_fv > 0 and bull_fv > 0 and (bull_fv / bear_fv) > 8:
            f.append("Bull/bear spread is {:.1f}× — wide scenario range; treat as directional indicator only".format(bull_fv / bear_fv))
        ni = d.get("net_income")
        if ni and rev and rev > 0 and ni / rev < 0:
            f.append("Company is currently unprofitable — ERG margin-expansion assumption is speculative")
        flags["ERG"] = f

    # ── New advanced models ────────────────────────────────────────────────
    monte = growth_results.get("monte_carlo_dcf") or next(
        (r for r in classic_results if r.get("method") == "Monte Carlo DCF"), None)
    if monte:
        f = []
        conv = monte.get("convergence_pct", 100)
        if conv < 70:
            f.append("Only {:.0f}% of Monte Carlo sims produced positive equity — high uncertainty.".format(conv))
        p10 = monte.get("p10", 0); p90 = monte.get("p90", 0)
        if p10 > 0 and p90 / p10 > 5:
            f.append("P90/P10 spread is {:.1f}× — extremely wide distribution; treat median as rough estimate.".format(p90/p10))
        flags["Monte Carlo DCF"] = f

    three = next((r for r in classic_results if r.get("method") == "Three-Stage DCF"), None)
    if three:
        f = []
        tp = three.get("details", {}).get("term_pct", 0)
        if tp > 85:
            f.append("Terminal value is {:.0f}% of EV — result dominated by far-future assumptions.".format(tp))
        if three.get("debt_heavy"):
            f.append("Net debt exceeds 80% of DCF enterprise value — equity residual unreliable.")
        flags["Three-Stage DCF"] = f

    rim_r = next((r for r in classic_results if r.get("method") == "RIM"), None)
    if rim_r:
        f = []
        roe_u = rim_r.get("roe_used")
        if roe_u is not None and roe_u < 0:
            f.append("ROE is negative — residual income model will produce misleading results.")
        if d.get("sector") in ("Financials", "Real Estate"):
            f.append("Financial sector: book value may include goodwill distortions; interpret with care.")
        flags["RIM"] = f

    roic_r = next((r for r in classic_results if r.get("method") == "ROIC Excess Return"), None)
    if roic_r:
        f = []
        if not roic_r.get("reliable"):
            f.append(roic_r.get("warning", "ROIC below WACC — company destroying economic value."))
        ic = roic_r.get("ic_per_share", 0)
        if ic <= 0:
            f.append("Invested capital per share is ≤ 0 — balance sheet structure makes this model unreliable.")
        flags["ROIC Excess Return"] = f

    fcfy = next((r for r in classic_results if r.get("method") == "FCF Yield"), None)
    if fcfy:
        f = []
        ni = d.get("net_income"); fcf = d.get("fcf")
        if ni and fcf and ni > 0:
            ratio = fcf / ni
            if ratio > 3.0 or ratio < 0.25:
                f.append("FCF/Net income ratio {:.1f}× suggests one-time items distorting cash flows.".format(ratio))
        flags["FCF Yield"] = f

    ddm_r = growth_results.get("ddm") or next(
        (r for r in classic_results if r.get("method") == "DDM"), None)
    if ddm_r:
        f = []
        if ddm_r.get("warning"):
            f.append(ddm_r["warning"])
        gS_dec = (ddm_r.get("gS") or 0) / 100
        if gS_dec < TERMINAL_GROWTH_RATE:
            f.append("Estimated growth ({:.1f}%) below terminal rate ({:.1f}%) — H-Model assumption violated.".format(
                gS_dec*100, TERMINAL_GROWTH_RATE*100))
        flags["DDM"] = f

    ncav_r = next((r for r in classic_results if r.get("method") == "NCAV"), None)
    if ncav_r:
        f = []
        if ncav_r.get("warning"):
            f.append(ncav_r["warning"])
        if d.get("sector") in ("Financials",):
            f.append("Financial sector: current assets include loans/receivables — NCAV interpretation differs.")
        flags["NCAV"] = f

    scurve = growth_results.get("scurve_tam")
    if scurve:
        f = []
        scen = scurve.get("scenarios", [])
        if len(scen) >= 2:
            bear_fv = scen[0].get("fv", 0); bull_fv = scen[-1].get("fv", 0)
            if bear_fv > 0 and bull_fv / bear_fv > 10:
                f.append("Bull/bear spread {:.0f}× — S-Curve scenarios are very wide.".format(bull_fv/bear_fv))
        flags["S-Curve TAM"] = f

    pie_r = growth_results.get("pie")
    if pie_r:
        f = []
        adj = pie_r.get("adjustment_factor", 1.0)
        if adj > 1.5 or adj < 0.5:
            f.append("Adjustment factor {:.2f}× is extreme — implied vs analyst growth diverge significantly.".format(adj))
        flags["PIE"] = f

    mr = next((r for r in classic_results if r.get("method") == "Mean Reversion"), None)
    if mr:
        f = []
        if mr.get("n_components", 0) < 2:
            f.append("Only 1 historical multiple available — estimate less reliable.")
        hp = mr.get("hist_pe_5y")
        if hp and hp > 60:
            f.append("5-year historical P/E of {:.0f}× may be distorted by loss years.".format(hp))
        flags["Mean Reversion"] = f

    bay = growth_results.get("bayesian")
    if bay:
        f = []
        n = bay.get("n_models_used", 0)
        if n < 5:
            f.append("Only {} models contributed — ensemble less robust.".format(n))
        flags["Bayesian Ensemble"] = f

    mf = growth_results.get("multifactor")
    if mf:
        f = []
        if mf.get("n_factors", 4) < 3:
            f.append("Fewer than 3 of 4 scoring factors available — Multi-Factor score less reliable.")
        flags["Multi-Factor"] = f



    # ── Build final output ─────────────────────────────────────────
    out = {}
    for method, method_flags in flags.items():
        out[method] = {
            "reliable": len(method_flags) == 0,
            "flags":    method_flags,
            "reason":   method_flags[0] if method_flags else None,
        }
    return out


# ── § 7  VALUE SCREENING ──────────────────────────────────────────────────────

# Value-trap exclusion thresholds (from valueScreener.py)
MAX_DEBT_EQUITY_NONFIN = 3.0    # D/E above this → flag (non-financial sectors)
MIN_ROE_QUALITY        = 0.15   # ROE >= 15% → "quality" company
MIN_OP_MARGIN          = 0.0    # operating margin must be positive to score

# Income / dividend sectors where yield matters
INCOME_SECTORS = {"Utilities", "Real Estate", "Consumer Staples", "Energy"}


def get_value_trap_flags(d):
    """Returns list of warning strings for value-trap conditions."""
    flags = []
    if d["fcf"] is not None and d["fcf"] < 0:
        flags.append("NEG FCF")
    if d["op_margin"] is not None and d["op_margin"] < 0:
        flags.append("NEG MARGIN")
    if (d["debt_equity"] is not None and
            d["debt_equity"] > MAX_DEBT_EQUITY_NONFIN and
            d["sector"] not in ("Financials", "Real Estate", "Utilities")):
        flags.append("HIGH DEBT")
    if d["rev_growth"] is not None and d["rev_growth"] < -5.0:
        flags.append("REV DECLINE")
    if d["eps"] is not None and d["eps"] < 0:
        flags.append("LOSING $")
    return flags


def compute_value_rank_score(discount, conviction, roe, pos52, rev_growth,
                              eps_growth, n_methods, trap_flags, div_yield, sector,
                              roic=None, gross_margin_v=None, buyback_pct=None):
    """
    0-100 opportunity score:
      35 pts — valuation discount magnitude
      15 pts — conviction (method agreement + quality gates)
      20 pts — quality: ROIC (12) + ROE (8)  [v3: ROIC now primary]
       8 pts — gross margin (pricing power / moat proxy)
       7 pts — momentum / 52-wk position
       5 pts — growth (revenue + EPS)
       5 pts — dividend yield (income) / buyback + growth (others)
    Deductions for value-trap flags (-8 pts each).
    """
    score = 0.0

    # 1. Discount (35 pts) — capped at 60%
    d_capped = max(-30.0, min(60.0, discount))
    score += max(0.0, d_capped / 60.0 * 35.0)

    # 2. Conviction (15 pts)
    conv_pts = {"HIGH": 15, "MED": 9, "LOW": 3}
    score += conv_pts.get(conviction, 3)

    # 3. ROIC — primary quality signal (12 pts)
    # Measures returns on ALL capital; the clearest sign a cheap stock is not a trap
    if roic is not None:
        if   roic >= 25.0: score += 12
        elif roic >= 15.0: score += 9
        elif roic >= 10.0: score += 6
        elif roic >= 5.0:  score += 3
        elif roic > 0.0:   score += 1
        # negative ROIC → 0 pts (destroying value)

    # 4. ROE — secondary quality signal (8 pts)
    if roe is not None:
        if   roe >= 30.0: score += 8
        elif roe >= 20.0: score += 6
        elif roe >= 15.0: score += 4
        elif roe >= 8.0:  score += 2
        elif roe >= 0.0:  score += 1

    # 5. Gross margin — pricing power / moat (8 pts)
    if gross_margin_v is not None:
        if   gross_margin_v >= 70.0: score += 8
        elif gross_margin_v >= 50.0: score += 6
        elif gross_margin_v >= 35.0: score += 4
        elif gross_margin_v >= 20.0: score += 2

    # 6. Momentum — 52-wk position (7 pts)
    # Sweet spot for value: pulled back from highs but not in free fall
    if pos52 is not None:
        if   0.15 <= pos52 <= 0.55: score += 7
        elif 0.10 <= pos52 <  0.15: score += 5
        elif 0.55 <  pos52 <= 0.75: score += 5
        elif pos52 < 0.10:          score += 2   # near 52wk low = caution
        else:                        score += 3   # near 52wk high

    # 7. Growth (5 pts)
    growth_pts = 0
    if rev_growth is not None:
        if   rev_growth >= 15.0: growth_pts += 3
        elif rev_growth >= 5.0:  growth_pts += 2
        elif rev_growth >= 0.0:  growth_pts += 1
    if eps_growth is not None:
        if   eps_growth >= 15.0: growth_pts += 2
        elif eps_growth >= 5.0:  growth_pts += 1
    score += min(5, growth_pts)

    # 8. Shareholder return / income bonus (5 pts)
    if sector in INCOME_SECTORS and div_yield is not None and div_yield > 0:
        if   div_yield >= 5.0: score += 5
        elif div_yield >= 3.5: score += 4
        elif div_yield >= 2.0: score += 3
        else:                  score += 1
    else:
        # Buyback signal: management buying back stock = cheap by their own assessment
        if buyback_pct is not None and buyback_pct > 0:
            if   buyback_pct >= 5.0: score += 5   # aggressive buyback
            elif buyback_pct >= 2.0: score += 3
            elif buyback_pct >= 0.5: score += 1
        elif rev_growth is not None and rev_growth >= 10.0:
            score += 2   # growth bonus if no buyback data

    # Deductions for trap flags (-8 pts each)
    score -= len(trap_flags) * 8

    return max(0.0, min(100.0, round(score, 1)))


def compute_peg(pe, growth):
    """Peter Lynch PEG = P/E ÷ (growth rate * 100). <1 = undervalued."""
    if pe and pe > 0 and growth and growth > 0:
        return round(pe / (growth * 100), 2)
    return None


# ── § 8  GROWTH SCREENING ─────────────────────────────────────────────────────

# Growth screener config constants (from growthScreener.py)
PEG_BEAR   = 0.75
PEG_BASE   = 1.0
PEG_BULL   = 1.5

PEG_WEIGHT = 0.60
REV_WEIGHT = 0.40

MAX_GROWTH_RATE   = 0.80
MAX_FWD_PE        = 45.0   # 45× cap prevents cyclical/small-cap PE inflation
MAX_PEG_GROWTH    = 40.0   # growth rate (%) used in PEG PE calc — capped lower than MAX_GROWTH_RATE
MIN_GROWTH_THRESH = 0.05

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


def get_growth_risk_flags(d):
    flags = []
    if d["eps_ttm"] is not None and d["eps_ttm"] <= 0:
        flags.append("UNPROFITABLE")
    if d["fcf"] is not None and d["fcf"] < 0:
        flags.append("NEG FCF")
    if d["op_margin"] is not None and d["op_margin"] < 0:
        flags.append("NEG MARGIN")
    if (d["de_ratio"] is not None and d["de_ratio"] > 3.0 and
            d["sector"] not in ("Financials", "Real Estate", "Utilities")):
        flags.append("HIGH DEBT")
    if d["rev_growth"] is not None and d["rev_growth"] < 0:
        flags.append("REV DECLINE")
    return flags


def calc_peg_targets(d):
    """
    Returns (bear, base, bull, growth_used, eps_used) price targets using Forward PEG method.

    Formula: Price Target = Forward EPS × min(Forward Growth% × PEG target, MAX_FWD_PE)

    Forward EPS priority:
      1. Analyst next-quarter estimate × 4 (annualised) — most accurate
      2. TTM EPS × blended forward growth rate
      3. FY EPS × blended forward growth rate

    Forward Growth Rate priority:
      1. Implied from annualised fwd EPS vs TTM EPS — derived from analyst estimates
      2. Blended forward growth (70/30 or 50/50 of quarterly vs TTM)
      3. Revenue growth as EPS proxy (×0.8 haircut)
    """
    fwd_eps = d["eps_fwd"]
    if not fwd_eps or fwd_eps <= 0:
        return None, None, None, None, None

    # Use the best available forward growth rate for PEG calculation
    # peg_growth_rate = implied from analyst fwd EPS if available, else blended
    growth_pct = d["peg_growth_rate"]
    if growth_pct is None and d["rev_growth"] is not None:
        growth_pct = d["rev_growth"] * 0.8   # revenue growth → proxy EPS growth
    if growth_pct is None or growth_pct <= 0:
        return None, None, None, None, None

    # Cap growth used in PE calc at MAX_PEG_GROWTH (40%) — beyond that the
    # market does not linearly reward growth with a higher P/E multiple.
    # Full growth_pct is preserved for bull scenario (PEG_BULL already stretches it).
    g = min(growth_pct, MAX_PEG_GROWTH)

    bear = fwd_eps * min(g * PEG_BEAR, MAX_FWD_PE)
    base = fwd_eps * min(g * PEG_BASE, MAX_FWD_PE)
    bull = fwd_eps * min(g * PEG_BULL, MAX_FWD_PE)

    if bear <= 0: return None, None, None, None, None
    return bear, base, bull, growth_pct, fwd_eps


def calc_rev_target(d):
    """
    Returns a price target via revenue extrapolation.
    1. Project forward revenue = TTM revenue × (1 + rev_growth)
    2. Apply net margin to get forward net income
    3. Apply sector P/E to get target market cap
    4. Divide by shares to get price target
    """
    fwd_rev = d["fwd_rev"]
    if not fwd_rev or fwd_rev <= 0:
        return None

    shares = d["shares"]
    if not shares or shares <= 0:
        return None

    # Net margin: use current, but also check if margins are expanding
    net_mar = d["net_margin"]
    op_mar  = d["op_margin"]

    # If no net margin but we have operating margin, estimate net margin
    if net_mar is None and op_mar is not None:
        net_mar = op_mar * 0.72   # rough tax/interest adjustment

    # For unprofitable companies, use FCF margin as proxy if available
    if (net_mar is None or net_mar <= 0) and d["fcf_margin"] is not None and d["fcf_margin"] > 0:
        net_mar = d["fcf_margin"]

    if net_mar is None or net_mar <= 0:
        return None

    # Forward net income estimate
    fwd_net_income = fwd_rev * (net_mar / 100.0)
    if fwd_net_income <= 0:
        return None

    # Sector P/E as valuation anchor
    sec_pe = SECTOR_PE.get(d["sector"], MARKET_PE)

    # Modest growth premium — forward revenue already bakes in the growth,
    # so layering a large P/E bonus double-counts it.
    rev_g = d["rev_growth_dec"] or 0
    if   rev_g >= 0.50: growth_pe_bonus = sec_pe * 0.25
    elif rev_g >= 0.25: growth_pe_bonus = sec_pe * 0.15
    elif rev_g >= 0.10: growth_pe_bonus = sec_pe * 0.07
    else:               growth_pe_bonus = 0.0

    applied_pe = min(sec_pe + growth_pe_bonus, MAX_FWD_PE)

    target_mktcap = fwd_net_income * applied_pe
    return target_mktcap / shares


def calc_ev_target(d):
    """
    Derives a fair share price by:
      1. Estimating forward EBITDA (TTM EBITDA × blended growth rate)
      2. Applying a sector EV/EBITDA multiple, adjusted upward for high growth
      3. Subtracting net debt to get equity value
      4. Dividing by shares outstanding

    Formula:
      Fair EV     = Forward EBITDA × growth-adjusted sector multiple
      Fair Equity = Fair EV − Net Debt
      Fair Price  = Fair Equity / Shares
    """
    ebitda = d.get("ebitda")
    if not ebitda or ebitda <= 0:
        return None

    shares = d.get("shares")
    if not shares or shares <= 0:
        return None

    net_debt = d.get("net_debt_v") or 0.0

    # Forward EBITDA — grow TTM EBITDA by best available growth rate
    ebitda_g = d.get("ebitda_growth")
    rev_g    = d.get("rev_growth")
    growth   = ebitda_g if ebitda_g is not None else rev_g
    if growth is not None:
        fwd_ebitda = ebitda * (1 + min(growth / 100.0, MAX_GROWTH_RATE))
    else:
        fwd_ebitda = ebitda

    # Sector base EV/EBITDA multiple
    sec_mult = SECTOR_EV_EBITDA.get(d.get("sector", ""), MARKET_EV_EBITDA)

    # Modest growth premium on EV/EBITDA — forward EBITDA already reflects
    # growth, so the multiple bonus should be conservative.
    g = growth or 0
    if   g >= 50: mult_bonus = sec_mult * 0.25
    elif g >= 25: mult_bonus = sec_mult * 0.15
    elif g >= 10: mult_bonus = sec_mult * 0.07
    else:         mult_bonus = 0.0

    applied_mult = min(sec_mult + mult_bonus, sec_mult * 1.75)

    fair_ev     = fwd_ebitda * applied_mult
    fair_equity = fair_ev - net_debt
    if fair_equity <= 0:
        return None

    return fair_equity / shares


def calc_quality_score(d):
    """
    Business quality — measures durability and efficiency of growth.
    ROIC is primary (replaces ROE as main quality signal).
    """
    score = 0.0

    # ROIC — best single quality metric (10 pts)
    # Measures how efficiently company generates returns on ALL invested capital
    roic = d.get("roic")
    if roic is not None:
        if   roic >= 40: score += 10  # exceptional (NVDA/AAPL territory)
        elif roic >= 25: score += 8
        elif roic >= 15: score += 6
        elif roic >= 8:  score += 3
        elif roic > 0:   score += 1

    # ROE — secondary quality signal (4 pts)
    roe = d.get("roe")
    if roe is not None and roe > 0:
        if   roe >= 30: score += 4
        elif roe >= 20: score += 3
        elif roe >= 15: score += 2
        elif roe >= 8:  score += 1

    # Gross margin — pricing power / moat (8 pts)
    gm = d.get("gross_margin")
    if gm is not None:
        if   gm >= 70: score += 8
        elif gm >= 55: score += 6
        elif gm >= 40: score += 4
        elif gm >= 25: score += 2

    # Operating margin — operational efficiency (5 pts)
    om = d.get("op_margin")
    if om is not None:
        if   om >= 25: score += 5
        elif om >= 15: score += 4
        elif om >= 8:  score += 2
        elif om >= 0:  score += 1

    # FCF generation (5 pts) — cash is king
    fcf_m = d.get("fcf_margin")
    fcf   = d.get("fcf")
    if fcf is not None and fcf > 0:
        if   fcf_m and fcf_m >= 20: score += 5
        elif fcf_m and fcf_m >= 10: score += 4
        else: score += 3
    elif fcf_m is not None and fcf_m > 0:
        score += 2

    # Liquidity gate — financial health (3 pts)
    cr = d.get("current_ratio")
    if cr is not None:
        if   cr >= 2.0: score += 3
        elif cr >= 1.5: score += 2
        elif cr >= 1.0: score += 1
        else:           score -= 2   # below 1 = potential liquidity risk

    return min(35.0, score)


def calc_growth_momentum_score(d):
    """
    Multi-dimensional growth assessment using all available growth signals.
    Rewards breadth (multiple growth axes) and acceleration.
    """
    score = 0.0

    # Revenue growth + acceleration (10 pts)
    rg = d.get("rev_growth")
    rg_qoq = d.get("rev_qoq")   # sequential QoQ growth
    if rg is not None:
        if   rg >= 50: score += 7
        elif rg >= 25: score += 6
        elif rg >= 15: score += 4
        elif rg >= 10: score += 2
        elif rg >= 5:  score += 1
    # Sequential acceleration bonus
    if rg_qoq is not None and rg_qoq > 5:
        score += min(3, rg_qoq / 10)  # up to +3 pts

    # EPS growth + acceleration (8 pts)
    eg = d.get("eps_growth")
    eg_qoq = d.get("eps_qoq")
    if eg is not None and eg > 0:
        if   eg >= 50: score += 5
        elif eg >= 25: score += 4
        elif eg >= 15: score += 3
        elif eg >= 5:  score += 1
    if eg_qoq is not None and eg_qoq > 5:
        score += min(3, eg_qoq / 10)

    # Gross profit growth — margin expansion signal (4 pts)
    gpg = d.get("gp_growth")
    if gpg is not None:
        if   gpg >= 40: score += 4
        elif gpg >= 20: score += 3
        elif gpg >= 10: score += 2
        elif gpg >= 5:  score += 1

    # FCF growth — cash conversion quality (5 pts)
    fcfg = d.get("fcf_growth")
    if fcfg is not None and fcfg > 0:
        if   fcfg >= 50: score += 5
        elif fcfg >= 25: score += 4
        elif fcfg >= 10: score += 2
        elif fcfg >= 0:  score += 1

    # Net income growth (4 pts)
    nig = d.get("ni_growth")
    if nig is not None and nig > 0:
        if   nig >= 50: score += 4
        elif nig >= 25: score += 3
        elif nig >= 10: score += 2
        elif nig >= 5:  score += 1

    # EBITDA growth — operational leverage (3 pts)
    ebg = d.get("ebitda_growth")
    if ebg is not None and ebg > 0:
        if   ebg >= 30: score += 3
        elif ebg >= 15: score += 2
        elif ebg >= 5:  score += 1

    # R&D investment — future pipeline (up to 3 pts, all sectors)
    # High R&D ratio = investing in future growth (rewarded when combined with
    # revenue growth; penalised if revenue is stagnant — spending without results)
    rd = d.get("rd_ratio")
    if rd is not None and rd > 0:
        rg_check = d.get("rev_growth") or 0
        if rg_check >= 10:    # growing AND investing = strong signal
            if   rd >= 15: score += 3
            elif rd >= 8:  score += 2
            elif rd >= 3:  score += 1
        elif rg_check >= 0:   # neutral growth + R&D = modest reward
            if rd >= 10: score += 1
        # If revenue declining, R&D doesn't help score

    return min(35.0, score)


def calc_momentum_score(d):
    """
    Technical and price momentum — market confirmation of the fundamental thesis.
    Upgraded with MoneyFlow, Stochastic, Williams %R, YTD + weekly perf.
    """
    score = 0.0

    # TradingView composite rating (4 pts) — most comprehensive single signal
    tv = d.get("tv_rating")
    if tv is not None:
        if   tv >= 0.5:  score += 4
        elif tv >= 0.1:  score += 2
        elif tv >= -0.1: score += 1

    # MoneyFlow (4 pts) — volume-weighted RSI, strongest volume/price signal
    mf = d.get("money_flow")
    if mf is not None:
        if   mf >= 70: score += 4   # strong buying pressure
        elif mf >= 55: score += 3
        elif mf >= 45: score += 1
        elif mf <= 30: score -= 2   # strong selling pressure

    # RSI (2 pts) — overbought/oversold
    rsi = d.get("rsi")
    if rsi is not None:
        if   50 <= rsi <= 65: score += 2   # healthy momentum zone
        elif 40 <= rsi < 50:  score += 1
        elif rsi > 75:        score -= 1   # overbought risk

    # Stochastic — entry timing (2 pts)
    sk = d.get("stoch_k")
    sd = d.get("stoch_d")
    if sk is not None and sd is not None:
        if 20 <= sk <= 80 and sk > sd:  score += 2  # bullish cross, not extreme
        elif sk < 20:                    score += 1  # oversold — potential entry

    # Williams %R — overbought/oversold (1 pt)
    wr = d.get("williams_r")
    if wr is not None:
        if   -50 <= wr <= -20: score += 1   # healthy momentum, not overbought
        elif wr > -20:         score -= 1   # overbought

    # Price momentum across timeframes (7 pts total)
    p1m = d.get("perf_1m")
    if p1m is not None:
        if   p1m >= 10: score += 2
        elif p1m >= 3:  score += 1
        elif p1m <= -10:score -= 1

    p3m = d.get("perf_3m")
    if p3m is not None:
        if   p3m >= 20: score += 3
        elif p3m >= 8:  score += 2
        elif p3m >= 0:  score += 1
        elif p3m <= -15:score -= 1

    p6m = d.get("perf_6m")
    if p6m is not None:
        if   p6m >= 30: score += 2
        elif p6m >= 10: score += 1

    # YTD performance context (bonus, not penalised to avoid double-counting)
    ytd = d.get("perf_ytd")
    if ytd is not None and ytd >= 20:
        score += 1   # outperforming YTD adds modest confirmation

    # 52-week position
    pos = d.get("pos52")
    if pos is not None:
        if   pos >= 0.85: score += 1
        elif pos <= 0.20: score -= 1

    return min(20.0, max(-5.0, score))


def calc_valuation_score(d, base_target):
    """
    Multi-lens valuation — PEG, P/FCF, EV/EBITDA, and upside to model target.
    Kept intentionally modest (10pts) — valuation alone shouldn't dominate.
    """
    score = 0.0
    price = d.get("price")

    # Upside to model target (4 pts)
    if price and price > 0 and base_target and base_target > 0:
        upside = (base_target - price) / price * 100
        if   upside >= 40: score += 4
        elif upside >= 20: score += 3
        elif upside >= 10: score += 2
        elif upside >= 0:  score += 1

    # Forward PEG (2 pts)
    pe = d.get("pe_ttm")
    eg = d.get("eps_growth")
    if pe and pe > 0 and eg and eg > 0:
        peg = pe / eg
        if   peg <= 0.75: score += 2
        elif peg <= 1.0:  score += 1.5
        elif peg <= 1.5:  score += 1

    # P/FCF (2 pts) — FCF-based valuation (harder to manipulate than EPS)
    p_fcf = d.get("p_fcf")
    if p_fcf is not None and p_fcf > 0:
        if   p_fcf <= 15: score += 2
        elif p_fcf <= 25: score += 1.5
        elif p_fcf <= 35: score += 1

    # EV/EBITDA (2 pts) — capital-structure-neutral valuation
    ev_eb = d.get("ev_ebitda")
    if ev_eb is not None and ev_eb > 0:
        if   ev_eb <= 12: score += 2
        elif ev_eb <= 20: score += 1.5
        elif ev_eb <= 30: score += 1

    return min(10.0, score)


def calc_growth_score(d, base_target, flags):
    """
    Four-pillar composite score:
      Pillar 1 — Quality         (0–35): ROIC, ROE, margins, FCF, liquidity
      Pillar 2 — Growth Momentum (0–35): revenue/EPS/FCF/GP/NI/EBITDA growth + R&D
      Pillar 3 — Technical       (0–20): TV rating, MoneyFlow, RSI, Stoch, price perf
      Pillar 4 — Valuation       (0–10): PEG, P/FCF, EV/EBITDA, model upside
    Total: 0–100, deductions for risk flags.
    """
    q  = calc_quality_score(d)             # 0–35
    gm = calc_growth_momentum_score(d)     # 0–35
    t  = calc_momentum_score(d)            # 0–20
    v  = calc_valuation_score(d, base_target)  # 0–10
    raw = q + gm + t + v

    # Risk flag deductions (4 pts each — meaningful but not catastrophic)
    raw -= len(flags) * 4

    return max(0.0, min(100.0, round(raw, 1)))


def derive_sentiment(d):
    """
    Compute a sentiment dict from already-fetched TradingView data.
    Returns keys: label, score (0-100), signals (list of strings)
    """
    score   = 50   # start neutral
    signals = []

    # ── TradingView composite rating (most weight) ─────────────────────────────
    tv = d.get("tv_rating")
    if tv is not None:
        tv_pts = tv * 25   # maps -1→-25, 0→0, +1→+25
        score += tv_pts
        if   tv >= 0.5:  signals.append("TV: Strong Buy")
        elif tv >= 0.1:  signals.append("TV: Buy")
        elif tv >= -0.1: signals.append("TV: Neutral")
        elif tv >= -0.5: signals.append("TV: Sell")
        else:            signals.append("TV: Strong Sell")

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = d.get("rsi")
    if rsi is not None:
        if rsi >= 70:
            score -= 8
            signals.append(f"RSI {rsi:.0f} (overbought)")
        elif rsi >= 55:
            score += 5
            signals.append(f"RSI {rsi:.0f} (bullish)")
        elif rsi <= 30:
            score += 8    # oversold = potential bounce
            signals.append(f"RSI {rsi:.0f} (oversold — potential bounce)")
        elif rsi <= 45:
            score -= 5
            signals.append(f"RSI {rsi:.0f} (bearish)")
        else:
            signals.append(f"RSI {rsi:.0f} (neutral)")

    # ── Price momentum ────────────────────────────────────────────────────────
    perf_1m = d.get("perf_1m")
    perf_3m = d.get("perf_3m")
    if perf_1m is not None:
        if   perf_1m >= 10:  score += 8;  signals.append(f"1M perf: +{perf_1m:.1f}% (strong)")
        elif perf_1m >= 3:   score += 4;  signals.append(f"1M perf: +{perf_1m:.1f}%")
        elif perf_1m <= -10: score -= 8;  signals.append(f"1M perf: {perf_1m:.1f}% (weak)")
        elif perf_1m <= -3:  score -= 4;  signals.append(f"1M perf: {perf_1m:.1f}%")
    if perf_3m is not None:
        if   perf_3m >= 20:  score += 6;  signals.append(f"3M perf: +{perf_3m:.1f}% (strong)")
        elif perf_3m >= 5:   score += 3
        elif perf_3m <= -20: score -= 6;  signals.append(f"3M perf: {perf_3m:.1f}% (weak)")
        elif perf_3m <= -5:  score -= 3

    # ── 52-week position ──────────────────────────────────────────────────────
    pos52 = d.get("pos52")
    if pos52 is not None:
        if   pos52 >= 0.85: score += 5;  signals.append(f"Near 52wk high ({pos52*100:.0f}%)")
        elif pos52 >= 0.70: score += 2
        elif pos52 <= 0.30: score -= 5;  signals.append(f"Near 52wk low ({pos52*100:.0f}%)")
        elif pos52 <= 0.50: score -= 2

    # ── EPS growth acceleration ───────────────────────────────────────────────
    eps_fq  = d.get("eps_growth_fq")
    eps_ttm = d.get("eps_growth_ttm")
    if eps_fq is not None and eps_ttm is not None:
        if eps_fq > eps_ttm + 10:
            score += 5
            signals.append(f"EPS accelerating ({eps_ttm:.0f}%→{eps_fq:.0f}%)")
        elif eps_fq < eps_ttm - 10:
            score -= 5
            signals.append(f"EPS decelerating ({eps_ttm:.0f}%→{eps_fq:.0f}%)")

    score = max(0, min(100, round(score)))

    if   score >= 70: label = "Bullish"
    elif score >= 55: label = "Leaning Bullish"
    elif score >= 45: label = "Neutral"
    elif score >= 30: label = "Leaning Bearish"
    else:             label = "Bearish"

    return dict(label=label, score=score, signals=signals)


def derive_accumulation(d):
    """
    Compute an accumulation/institutional buying proxy score (0-100).
    Returns dict with: label, score, signals.
    """
    score   = 50
    signals = []

    # ── 1. MACD histogram ─────────────────────────────────────────────────────
    macd_hist   = d.get("macd_hist")
    macd_line   = d.get("macd_line")
    macd_signal = d.get("macd_signal")
    if macd_hist is not None:
        if macd_hist > 0 and macd_line is not None and macd_signal is not None and macd_line > macd_signal:
            score += 15
            signals.append(f"MACD bullish crossover (hist={macd_hist:+.3f})")
        elif macd_hist > 0:
            score += 8
            signals.append(f"MACD hist positive ({macd_hist:+.3f})")
        elif macd_hist < 0 and macd_line is not None and macd_signal is not None and macd_line < macd_signal:
            score -= 15
            signals.append(f"MACD bearish crossover (hist={macd_hist:+.3f})")
        else:
            score -= 8
            signals.append(f"MACD hist negative ({macd_hist:+.3f})")

    # ── 2. Momentum ───────────────────────────────────────────────────────────
    mom = d.get("mom")
    if mom is not None:
        if   mom > 10:  score += 10; signals.append(f"Momentum strong ({mom:+.1f})")
        elif mom > 0:   score += 5;  signals.append(f"Momentum positive ({mom:+.1f})")
        elif mom < -10: score -= 10; signals.append(f"Momentum weak ({mom:+.1f})")
        else:           score -= 5;  signals.append(f"Momentum negative ({mom:+.1f})")

    # ── 3. Volume trend ───────────────────────────────────────────────────────
    rel_vol     = d.get("rel_vol")
    avg_vol_30d = d.get("avg_vol_30d")
    avg_vol_90d = d.get("avg_vol_90d")

    if rel_vol is not None:
        if   rel_vol >= 1.5: score += 12; signals.append(f"Volume surge ({rel_vol:.1f}x avg — strong accumulation signal)")
        elif rel_vol >= 1.15: score += 6; signals.append(f"Above-avg volume ({rel_vol:.1f}x)")
        elif rel_vol <= 0.7:  score -= 6; signals.append(f"Below-avg volume ({rel_vol:.1f}x — distribution risk)")

    # Volume trend: is 30d avg growing vs 90d avg? (expanding participation)
    if avg_vol_30d and avg_vol_90d and avg_vol_90d > 0:
        vol_trend = (avg_vol_30d - avg_vol_90d) / avg_vol_90d * 100
        if   vol_trend >= 15: score += 8;  signals.append(f"Volume expanding vs 90d avg (+{vol_trend:.0f}%)")
        elif vol_trend >= 5:  score += 4
        elif vol_trend <= -15: score -= 8; signals.append(f"Volume contracting vs 90d avg ({vol_trend:.0f}%)")
        elif vol_trend <= -5:  score -= 4

    # ── 4. ADX + Directional indicators ──────────────────────────────────────
    adx       = d.get("adx")
    adx_plus  = d.get("adx_plus")
    adx_minus = d.get("adx_minus")
    if adx is not None and adx_plus is not None and adx_minus is not None:
        if adx >= 25 and adx_plus > adx_minus:
            score += 12
            signals.append(f"Strong uptrend: ADX={adx:.1f}, +DI={adx_plus:.1f} > -DI={adx_minus:.1f}")
        elif adx >= 25 and adx_minus > adx_plus:
            score -= 12
            signals.append(f"Strong downtrend: ADX={adx:.1f}, -DI={adx_minus:.1f} > +DI={adx_plus:.1f}")
        elif adx_plus > adx_minus:
            score += 5
            signals.append(f"Mild buying pressure (+DI={adx_plus:.1f} > -DI={adx_minus:.1f})")
        else:
            score -= 5
            signals.append(f"Mild selling pressure (-DI={adx_minus:.1f} > +DI={adx_plus:.1f})")

    # ── 5. MoneyFlow (volume-weighted RSI — strongest vol/price signal) ─────
    mf = d.get("money_flow")
    if mf is not None:
        if   mf >= 70: score += 10; signals.append(f"MoneyFlow {mf:.0f} — strong institutional buying")
        elif mf >= 55: score += 6;  signals.append(f"MoneyFlow {mf:.0f} — buying pressure")
        elif mf >= 45: score += 2;  signals.append(f"MoneyFlow {mf:.0f} — neutral")
        elif mf <= 30: score -= 10; signals.append(f"MoneyFlow {mf:.0f} — strong selling pressure")
        else:          score -= 4;  signals.append(f"MoneyFlow {mf:.0f} — mild selling")

    score = max(0, min(100, round(score)))

    if   score >= 75: label = "Strong Accumulation"
    elif score >= 60: label = "Accumulation"
    elif score >= 45: label = "Neutral"
    elif score >= 30: label = "Distribution"
    else:             label = "Strong Distribution"

    return dict(label=label, score=score, signals=signals)


# ── § 9  ADVANCED CLASSIC MODELS ──────────────────────────────────────────────

def run_three_stage_dcf(d: dict) -> dict:
    """
    Three-stage 12-year FCF DCF:
      Stage 1 (years 1-3):   full est_growth
      Stage 2 (years 4-7):   linear decay from est_growth to 2×TERMINAL_GROWTH_RATE
      Stage 3 (years 8-12):  linear decay from 2×TERMINAL_GROWTH_RATE to TERMINAL_GROWTH_RATE
    Terminal value: Gordon Growth Model at TERMINAL_GROWTH_RATE

    More realistic than 2-stage for companies mid-way through their growth arc.
    Returns same dict shape as run_dcf.
    """
    fcf    = d.get("fcf")
    shares = d.get("shares")
    nd     = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    g      = d.get("est_growth") or 0.05

    if not (fcf and fcf > 0 and shares and shares > 0):
        return {
            "method": "Three-Stage DCF", "fair_value": None, "mos_value": None,
            "reliable": False, "debt_heavy": False, "warning": "Insufficient data",
        }

    wacc, wacc_source = calculate_wacc(d)

    pvs = []
    cf  = fcf

    # Stage 1: years 1-3, full growth
    for yr in range(1, 4):
        cf = cf * (1 + g)
        pvs.append(cf / (1 + wacc) ** yr)
    s1_exit_cf = cf

    # Stage 2: years 4-7, linear decay from g to 2*tg
    mid_g = TERMINAL_GROWTH_RATE * 2
    for yr in range(4, 8):
        t  = (yr - 3) / 4          # 0.25 → 1.0
        g2 = g * (1 - t) + mid_g * t
        cf = cf * (1 + g2)
        pvs.append(cf / (1 + wacc) ** yr)

    # Stage 3: years 8-12, linear decay from 2*tg to tg
    for yr in range(8, 13):
        t  = (yr - 7) / 5          # 0.2 → 1.0
        g3 = mid_g * (1 - t) + TERMINAL_GROWTH_RATE * t
        cf = cf * (1 + g3)
        pvs.append(cf / (1 + wacc) ** yr)

    sum_pv  = sum(pvs)
    term_val = cf * (1 + TERMINAL_GROWTH_RATE) / (wacc - TERMINAL_GROWTH_RATE)
    pv_term  = term_val / (1 + wacc) ** 12
    ev       = sum_pv + pv_term
    eq_val   = ev - nd

    debt_heavy = nd > ev * 0.8
    reliable   = eq_val > 0

    if eq_val <= 0:
        iv      = d.get("price", 0.0)
        warning = "DEBT-HEAVY: Net debt exceeds DCF enterprise value."
    else:
        iv      = round(eq_val / shares, 2)
        warning = "Debt load is significant — treat DCF with caution." if debt_heavy else None

    return {
        "method":      "Three-Stage DCF",
        "fair_value":  iv,
        "mos_value":   iv * (1 - MARGIN_OF_SAFETY),
        "wacc":        wacc,
        "wacc_source": wacc_source,
        "growth":      g,
        "reliable":    reliable,
        "debt_heavy":  debt_heavy,
        "warning":     warning,
        "net_debt":    nd,
        "details": {
            "sum_pv":   sum_pv,
            "pv_term":  pv_term,
            "ev":       ev,
            "eq_val":   max(eq_val, 0.0),
            "term_pct": (pv_term / ev * 100) if ev > 0 else 0,
        },
    }


def run_monte_carlo_dcf(d: dict, n_sims: int = 5000) -> dict:
    """
    Monte Carlo DCF — probabilistic 10-year 2-stage model.

    Samples three inputs from triangular distributions each simulation:
      g    ~ Triangular(low=est_growth×0.4, mode=est_growth, high=est_growth×1.6)
      wacc ~ Triangular(low=base_wacc×0.8,  mode=base_wacc,  high=base_wacc×1.2)
      tg   ~ Triangular(low=0.010,           mode=TERMINAL_GROWTH_RATE, high=0.045)

    Uses same 10-year two-stage growth-decay logic as run_dcf for each simulation.
    Returns median fair value plus confidence interval percentiles.
    Requires numpy; returns None if numpy unavailable or FCF ≤ 0.
    """
    if not _NUMPY_AVAIL:
        return None

    fcf    = d.get("fcf")
    shares = d.get("shares")
    nd     = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    g_base = d.get("est_growth") or 0.05

    if not (fcf and fcf > 0 and shares and shares > 0):
        return None

    base_wacc, wacc_source = calculate_wacc(d)

    # Draw samples
    rng    = np.random.default_rng(seed=42)
    g_arr  = rng.triangular(g_base * 0.4, g_base, g_base * 1.6, n_sims)
    w_arr  = rng.triangular(base_wacc * 0.8, base_wacc, base_wacc * 1.2, n_sims)
    tg_arr = rng.triangular(0.010, TERMINAL_GROWTH_RATE, 0.045, n_sims)
    # Clamp WACC and terminal growth to sane ranges
    w_arr  = np.clip(w_arr,  0.05, 0.20)
    tg_arr = np.clip(tg_arr, 0.005, 0.05)
    # Ensure wacc > tg for terminal value formula
    valid  = w_arr > tg_arr + 0.005
    g_arr  = g_arr[valid]; w_arr = w_arr[valid]; tg_arr = tg_arr[valid]
    n_valid = len(g_arr)
    if n_valid < 100:
        return None

    fvs = np.empty(n_valid)
    for i in range(n_valid):
        g1  = g_arr[i]
        wc  = w_arr[i]
        tg  = tg_arr[i]
        cf  = fcf
        pv  = 0.0
        for yr in range(1, 6):
            g1 = g1 * (1 - DECAY_RATE)
            g1 = max(g1, tg * 2)
            cf = cf * (1 + g1)
            pv += cf / (1 + wc) ** yr
        s1_g = g1
        for yr in range(6, 11):
            t   = (yr - 5) / 5
            g2  = s1_g * (1 - t) + tg * t
            cf  = cf * (1 + g2)
            pv += cf / (1 + wc) ** yr
        tv   = cf * (1 + tg) / (wc - tg)
        pv  += tv / (1 + wc) ** 10
        eq   = pv - nd
        fvs[i] = (eq / shares) if eq > 0 else 0.0

    positive = fvs[fvs > 0]
    conv_pct = len(positive) / n_valid * 100

    if len(positive) < 10:
        return None

    median_fv = float(np.median(positive))
    p10 = float(np.percentile(positive, 10))
    p25 = float(np.percentile(positive, 25))
    p75 = float(np.percentile(positive, 75))
    p90 = float(np.percentile(positive, 90))

    return {
        "method":          "Monte Carlo DCF",
        "fair_value":      round(median_fv, 2),
        "mos_value":       round(p25 * 0.80, 2),
        "p10":             round(p10, 2),
        "p25":             round(p25, 2),
        "p75":             round(p75, 2),
        "p90":             round(p90, 2),
        "n_sims":          n_valid,
        "std_dev":         round(float(np.std(positive)), 2),
        "convergence_pct": round(conv_pct, 1),
        "wacc":            base_wacc,
        "wacc_source":     wacc_source,
        "growth":          g_base,
        "reliable":        conv_pct >= 70,
        "debt_heavy":      nd > 0,
        "warning":         None if conv_pct >= 70 else
                           f"Only {conv_pct:.0f}% of simulations produced positive equity value.",
    }


def run_fcf_yield(d: dict) -> dict:
    """
    FCF Yield Intrinsic Value — bond-like floor valuation.

    Fair Value = FCF per share / target_yield
    where target_yield = RISK_FREE_RATE + 0.04  (equity risk spread)

    Analogous to pricing a bond: the stock is 'worth' the price at which
    its FCF yield equals the required equity yield.  Provides a conservative
    floor; most useful alongside DCF and multiple-based models.

    Returns None if FCF per share ≤ 0.
    """
    fcf_ps = d.get("fcf_per_share")
    price  = d.get("price") or 0.0

    if not (fcf_ps and fcf_ps > 0):
        return None

    target_yield  = RISK_FREE_RATE + 0.04
    fair_value    = round(fcf_ps / target_yield, 2)
    current_yield = (fcf_ps / price) if price > 0 else None
    yield_spread  = (current_yield - RISK_FREE_RATE) if current_yield is not None else None

    return {
        "method":        "FCF Yield",
        "fair_value":    fair_value,
        "mos_value":     round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "fcf_per_share": fcf_ps,
        "target_yield":  round(target_yield * 100, 2),
        "current_yield": round(current_yield * 100, 2) if current_yield else None,
        "yield_spread":  round(yield_spread * 100, 2) if yield_spread else None,
        "reliable":      True,
        "warning":       None,
    }


def run_rim(d: dict) -> dict:
    """
    Residual Income Model (Edwards-Bell-Ohlson / RIM):
      Value = BV_0 + sum_t[ RI_t / (1+Ke)^t ] + TV
    where:
      RI_t = EPS_t - (Ke × BV_{t-1})        residual income
      BV_t = BV_{t-1} + EPS_t × (1-payout)  book value grows by retained earnings
      TV   = RI_10 / (Ke - TERMINAL_GROWTH_RATE)  perpetuity at year 10

    EPS is projected at est_growth with 15% annual decay (same as run_dcf).
    Particularly useful for financials and asset-heavy businesses where
    FCF-based models are unreliable.

    Requires d["ext"]["book_value_ps"] > 0 and eps > 0.
    """
    ext        = d.get("ext") or {}
    book_ps    = ext.get("book_value_ps")
    eps        = d.get("eps")
    growth     = d.get("est_growth") or 0.05
    divs_ps    = ext.get("dividends_per_share") or 0.0
    beta       = d.get("beta") or 1.0

    if not (book_ps and book_ps > 0 and eps and eps > 0):
        return None

    ke     = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM  # CAPM cost of equity
    payout = min(0.95, max(0.0, divs_ps / eps)) if eps > 0 else 0.0
    DECAY  = 0.15

    bv     = book_ps
    g      = growth
    pvs    = []

    for yr in range(1, 11):
        g     = max(g * (1 - DECAY), TERMINAL_GROWTH_RATE)
        eps_t = eps * (1 + g) ** yr   # approximate — grows at decaying rate
        ri_t  = eps_t - (ke * bv)
        pvs.append(ri_t / (1 + ke) ** yr)
        bv   += eps_t * (1 - payout)  # book value grows by retained earnings

    # Terminal value: residual income perpetuity
    ri_terminal = pvs[-1] * (1 + ke) ** 10   # "un-discount" yr10 to get RI_10
    tv          = ri_terminal / (ke - TERMINAL_GROWTH_RATE) if ke > TERMINAL_GROWTH_RATE else 0
    pv_tv       = tv / (1 + ke) ** 10

    sum_pv     = sum(pvs)
    fair_value = book_ps + sum_pv + pv_tv
    fair_value = max(0.0, round(fair_value, 2))

    return {
        "method":          "RIM",
        "fair_value":      fair_value,
        "mos_value":       round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "book_value_ps":   book_ps,
        "roe_used":        round(eps / book_ps * 100, 1) if book_ps > 0 else None,
        "cost_of_equity":  round(ke * 100, 2),
        "payout_ratio":    round(payout * 100, 1),
        "pv_residual":     round(sum_pv, 2),
        "pv_terminal":     round(pv_tv, 2),
        "reliable":        fair_value > 0,
        "warning":         None,
    }


def run_roic_excess_return(d: dict) -> dict:
    """
    ROIC-Based Excess Return Model (McKinsey Valuation):
      Value = IC_ps × (1 + (ROIC - WACC) / (WACC - g))

    where IC_ps = (total_debt + stockholders_equity - cash) / shares

    Directly ties intrinsic value to economic value creation:
      - ROIC > WACC: each dollar of invested capital is worth >$1 → premium to book
      - ROIC < WACC: value destruction → fair value below book (flagged as warning)

    Requires d["ext"]["roic"] and d["ext"]["stockholders_equity"].
    """
    ext    = d.get("ext") or {}
    roic   = ext.get("roic")        # decimal, e.g. 0.25 for 25%
    equity = ext.get("stockholders_equity")
    shares = d.get("shares")
    debt   = d.get("total_debt") or 0.0
    cash   = d.get("cash") or 0.0
    g      = d.get("est_growth") or 0.05

    if not (roic is not None and equity is not None and shares and shares > 0):
        return None

    wacc, wacc_source = calculate_wacc(d)

    ic          = debt + equity - cash
    ic_ps       = ic / shares if shares > 0 else 0.0
    spread      = roic - wacc
    denom       = wacc - TERMINAL_GROWTH_RATE

    if denom <= 0:
        denom = 0.001   # safety guard — shouldn't happen after WACC clamp

    value_driver = spread / denom
    fair_value   = max(0.0, round(ic_ps * (1 + value_driver), 2))

    destroying_value = roic < wacc

    return {
        "method":        "ROIC Excess Return",
        "fair_value":    fair_value,
        "mos_value":     round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "roic":          round(roic * 100, 2),
        "wacc":          wacc,
        "wacc_source":   wacc_source,
        "ic_per_share":  round(ic_ps, 2),
        "spread":        round(spread * 100, 2),        # pct
        "value_driver":  round(value_driver, 3),
        "reliable":      not destroying_value,
        "warning":       (
            "ROIC ({:.1f}%) < WACC ({:.1f}%) — company destroying economic value; "
            "fair value below book.".format(roic*100, wacc*100)
        ) if destroying_value else None,
    }


def run_ddm_hmodel(d: dict) -> dict:
    """
    Dividend Discount Model — H-Model variant:
      V = D0×(1+gL)/(r-gL) + D0×H×(gS-gL)/(r-gL)

    Components:
      Gordon component  = D0×(1+gL)/(r-gL)  — value assuming long-run growth forever
      H-Model component = D0×H×(gS-gL)/(r-gL) — premium for supernormal growth period
    Parameters:
      H  = 5  (half-life of supernormal growth period, in years)
      gS = est_growth   (supernormal / near-term growth rate)
      gL = TERMINAL_GROWTH_RATE  (long-run sustainable growth)
      r  = CAPM cost of equity

    Most applicable to mature dividend-paying companies (utilities, REITs, staples).
    Requires d["ext"]["dividends_per_share"] > 0.
    """
    ext    = d.get("ext") or {}
    D0     = ext.get("dividends_per_share")
    beta   = d.get("beta") or 1.0
    gS     = d.get("est_growth") or 0.05
    gL     = TERMINAL_GROWTH_RATE
    H      = 5.0
    r      = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

    if not (D0 and D0 > 0):
        return None
    if r <= gL:
        r = gL + 0.02   # safety floor — cost of equity must exceed terminal growth

    denom            = r - gL
    gordon_component = D0 * (1 + gL) / denom
    hmodel_component = D0 * H * (gS - gL) / denom
    fair_value       = max(0.0, round(gordon_component + hmodel_component, 2))

    # Sustainability check
    eps        = d.get("eps") or 0
    payout_ok  = (D0 / eps <= 1.0) if eps > 0 else True
    warning    = None
    if eps > 0 and D0 / eps > 1.0:
        warning = "Payout ratio {:.0f}% exceeds 100% — dividend may be unsustainable.".format(D0/eps*100)

    return {
        "method":            "DDM",
        "fair_value":        fair_value,
        "mos_value":         round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "dividends_per_share": D0,
        "gS":                round(gS * 100, 1),
        "gL":                round(gL * 100, 1),
        "H":                 H,
        "cost_of_equity":    round(r * 100, 2),
        "gordon_component":  round(gordon_component, 2),
        "hmodel_component":  round(hmodel_component, 2),
        "reliable":          payout_ok,
        "warning":           warning,
    }


def run_ncav(d: dict) -> dict:
    """
    Graham Net Current Asset Value (NCAV):
      NCAV_ps = (current_assets - total_liabilities) / shares

    This is Benjamin Graham's liquidation floor — if the stock trades
    below NCAV, you are effectively buying the net current assets for
    free and getting long-term assets (PP&E, goodwill) thrown in.

    Not a growth model — most applicable for deep-value, asset-heavy
    or distressed situations.  Avoid interpreting it as a price target
    for growth companies.

    Requires d["ext"]["total_current_assets"] and d["ext"]["total_liabilities"].
    """
    ext      = d.get("ext") or {}
    cur_assets = ext.get("total_current_assets")
    tot_liab   = ext.get("total_liabilities")
    shares     = d.get("shares")
    price      = d.get("price") or 0.0

    if not (cur_assets is not None and tot_liab is not None and shares and shares > 0):
        return None

    ncav_total = cur_assets - tot_liab
    ncav_ps    = ncav_total / shares

    # NCAV is only a valid price target when positive.
    # Negative NCAV means total liabilities exceed current assets — not a floor,
    # just not applicable as a target.  Return None so the caller can log a
    # descriptive reason for the All Methods "N/A" section.
    if ncav_ps <= 0:
        return None

    fair_value = round(ncav_ps, 2)
    vs_price_discount = ((ncav_ps - price) / price * 100) if price > 0 else None

    if price > 0 and price < ncav_ps:
        graham_verdict = "Deep Value: trading below liquidation value"
    else:
        graham_verdict = "Premium to liquidation value"

    return {
        "method":               "NCAV",
        "fair_value":           fair_value,
        "mos_value":            round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "current_assets":       cur_assets,
        "total_liabilities":    tot_liab,
        "ncav_total":           round(ncav_total, 0),
        "ncav_ps":              round(ncav_ps, 2),
        "vs_price_discount_pct": round(vs_price_discount, 1) if vs_price_discount else None,
        "graham_verdict":       graham_verdict,
        "reliable":             True,
        "warning":              None,
    }


# ── § 10  ADVANCED GROWTH MODELS ─────────────────────────────────────────────

def run_scurve_tam(d: dict) -> dict:
    """
    S-Curve (Logistic) TAM Capture Model.

    Instead of exponential growth, models revenue following a logistic
    S-curve toward TAM saturation:
        R(t) = TAM / (1 + exp(-k × (t - t0)))

    Parameters derived from current position on the S-curve:
        current_share = revenue / TAM
        t0 = inflection point (half-TAM) estimated from current growth
        k  = -ln(1/current_share - 1) / (0 - t0)   (logistic parameter)

    Three scenarios vary the k parameter (±30%) to model different
    speed-of-adoption outcomes.

    Returns None if:
        - revenue or shares missing
        - current_share ≥ 0.70 (company already dominates TAM — logistic invalid)
        - current_share ≤ 0.001 (too early stage — t0 undefined)
    """
    rev     = d.get("revenue")
    growth  = d.get("est_growth") or 0.10
    gm      = d.get("gross_margin") or 50.0
    ni      = d.get("net_income")
    nd      = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    shares  = d.get("shares")
    price   = d.get("price") or 0.0

    if not (rev and rev > 0 and shares and shares > 0):
        return None

    wacc, _ = calculate_wacc(d)

    # TAM estimation (same formula as run_tam_scenario)
    tam_m = min(10 + max(0, (growth - 0.20) / 0.10 * 2.5), 20.0)
    tam   = rev * tam_m

    current_share = rev / tam
    if current_share >= 0.70 or current_share <= 0.001:
        return None

    # Inflection point: years to reach 50% TAM at current growth rate
    # Solve: rev*(1+growth)^t0 = tam/2 → t0 = log(tam/2/rev) / log(1+growth)
    try:
        t0 = math.log(tam / 2.0 / rev) / math.log(1 + max(growth, 0.01))
    except (ValueError, ZeroDivisionError):
        return None

    if t0 <= 0:
        t0 = 5.0  # fallback for already-past-inflection companies

    # Logistic parameter k
    try:
        k = -math.log(1.0 / current_share - 1.0) / (0.0 - t0)
    except (ValueError, ZeroDivisionError):
        return None

    if k <= 0:
        return None

    # Net margin — same logic as run_tam_scenario
    if ni and rev > 0 and ni / rev > 0.03:
        base_nm = max(0.05, min(0.60, ni / rev))
    else:
        gm_frac = gm / 100.0
        ratio   = 0.45 if gm_frac >= 0.70 else (0.38 if gm_frac >= 0.50 else 0.30)
        base_nm = max(0.05, gm_frac * ratio)

    scenarios = []
    for label, k_mult, mm in [("Bear", 0.70, 0.85), ("Base", 1.00, 1.00), ("Bull", 1.30, 1.15)]:
        k_s    = k * k_mult
        yr5_rev = tam / (1 + math.exp(-k_s * (5 - t0)))
        yr5_earn = yr5_rev * base_nm * mm
        pv       = yr5_earn * TERM_PE / (1 + wacc) ** 5
        fv       = max(0.0, pv / shares)
        upside   = ((fv - price) / price * 100) if price else 0
        scenarios.append({
            "label": label, "k_mult": k_mult, "yr5_rev": yr5_rev,
            "yr5_earn": yr5_earn, "fv": fv, "upside": upside,
        })

    base_fv = scenarios[1]["fv"]
    return {
        "method":         "S-Curve TAM",
        "fair_value":     round(base_fv, 2),
        "mos_value":      round(base_fv * (1 - MARGIN_OF_SAFETY), 2),
        "tam_est":        tam,
        "tam_mult":       round(tam_m, 1),
        "k_param":        round(k, 4),
        "t0_param":       round(t0, 1),
        "current_share_pct": round(current_share * 100, 2),
        "base_net_margin": round(base_nm * 100, 1),
        "wacc":           wacc,
        "scenarios":      scenarios,
        "reliable":       True,
        "warning":        None,
    }


def run_pie(d: dict) -> dict:
    """
    Price Implied Expectations (PIE).

    Reverse-engineers the revenue growth rate that the current market
    price implies, using a simplified 5-year DCF framework.  Compares
    the implied growth to analyst consensus (est_growth) to produce an
    adjustment factor:

        adjustment_factor = sqrt(est_growth / implied_growth)
        fair_value        = price × adjustment_factor

    Interpretation:
        implied > analyst  → market is more optimistic → fair_value < price (overvalued)
        implied < analyst  → market more pessimistic  → fair_value > price (upside)
        implied ≈ analyst  → fairly priced

    Also estimates implied operating margin by checking what margin is
    needed at the implied growth to justify the price.

    Returns None if price, revenue, or shares are missing.
    """
    price  = d.get("price")
    rev    = d.get("revenue")
    shares = d.get("shares")
    nd     = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    growth = d.get("est_growth") or 0.10
    op_mar = d.get("op_margin")   # percent

    if not (price and price > 0 and rev and rev > 0 and shares and shares > 0):
        return None

    wacc, _ = calculate_wacc(d)

    def dcf_at_growth(g):
        """Simple 5-year revenue-based DCF: projects revenue, applies a rough margin."""
        if g < -0.5:
            return 0.0
        # Use a simple "earnings yield" proxy: 10% mature net margin
        MATURE_NM = 0.10
        fv_arr = []
        for yr in range(1, 6):
            yr_rev  = rev * (1 + g) ** yr
            yr_earn = yr_rev * MATURE_NM
            fv_arr.append(yr_earn / (1 + wacc) ** yr)
        yr5_rev  = rev * (1 + g) ** 5
        terminal = yr5_rev * MATURE_NM * TERM_PE_PROXY / (1 + wacc) ** 5
        eq       = sum(fv_arr) + terminal - nd
        return (eq / shares) if eq > 0 else 0.0

    # Binary search for implied growth
    lo, hi = -0.20, 2.00
    for _ in range(60):
        mid = (lo + hi) / 2
        if dcf_at_growth(mid) < price:
            lo = mid
        else:
            hi = mid
    implied_growth = round((lo + hi) / 2, 4)

    # Adjustment factor — capped to avoid extreme outputs
    if implied_growth > 0.001 and growth > 0.001:
        raw_adj = (growth / implied_growth) ** 0.5
        adj     = round(max(0.30, min(3.0, raw_adj)), 4)
    elif implied_growth <= 0:
        adj = 2.0   # market implies decline; analyst thinks growth → big discount
    else:
        adj = 1.0

    fair_value = round(price * adj, 2)

    # Implied operating margin (at analyst growth)
    mkt_cap  = (d.get("market_cap") or 0)
    impl_op_margin = None
    if rev > 0 and mkt_cap > 0:
        # Rough: mkt_cap / (rev_yr5 × TERM_PE) ≈ what margin market assumes
        rev_yr5 = rev * (1 + growth) ** 5
        if rev_yr5 > 0:
            impl_earn = (mkt_cap / TERM_PE_PROXY) * (1 + wacc) ** 5
            impl_op_margin = round(impl_earn / rev_yr5 * 100, 1)

    growth_gap = round((growth - implied_growth) / max(abs(implied_growth), 0.01) * 100, 1)

    if abs(growth_gap) <= 15:
        verdict = "Aligned: market expectations close to analyst consensus"
    elif growth > implied_growth:
        verdict = "Market too pessimistic: analyst expects {:.0f}% growth, market implies {:.0f}%".format(
            growth*100, implied_growth*100)
    else:
        verdict = "Market too optimistic: market implies {:.0f}% growth vs analyst {:.0f}%".format(
            implied_growth*100, growth*100)

    return {
        "method":               "PIE",
        "fair_value":           fair_value,
        "mos_value":            round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "implied_revenue_growth": implied_growth,
        "analyst_growth":       growth,
        "implied_op_margin":    impl_op_margin,
        "consensus_op_margin":  round(op_mar, 1) if op_mar is not None else None,
        "growth_gap_pct":       growth_gap,
        "adjustment_factor":    adj,
        "verdict":              verdict,
        "reliable":             0.30 < adj < 2.50,
        "warning":              "Extreme adjustment factor — PIE inputs may not suit this stock." if not (0.30 < adj < 2.50) else None,
    }


def run_mean_reversion(d: dict) -> dict:
    """
    Mean Reversion to Historical Multiples.

    Fair value = average of up to 3 historically-anchored estimates:
        1. EPS   × hist_pe_5y
        2. FCF/share × hist_pfcf_5y
        3. (EBITDA × hist_eveb_5y − net_debt) / shares

    5-year median multiples are fetched via yfinance (stored in d["ext"]).
    Requires at least 1 component to be computable; returns None otherwise.

    Best for: mature, stable companies with predictable earnings.
    Less useful for: high-growth companies where historical multiples
    undervalue future potential.
    """
    ext     = d.get("ext") or {}
    hist_pe   = ext.get("hist_pe_5y")
    hist_pfcf = ext.get("hist_pfcf_5y")
    hist_eveb = ext.get("hist_eveb_5y")

    eps    = d.get("eps")
    fcf_ps = d.get("fcf_per_share")
    ebitda = d.get("ebitda")
    nd     = (d.get("total_debt") or 0.0) - (d.get("cash") or 0.0)
    shares = d.get("shares")

    components = []

    if hist_pe and hist_pe > 0 and eps and eps > 0:
        fv1 = round(eps * hist_pe, 2)
        components.append({"name": "P/E", "multiple": hist_pe, "fv": fv1})

    if hist_pfcf and hist_pfcf > 0 and fcf_ps and fcf_ps > 0:
        fv2 = round(fcf_ps * hist_pfcf, 2)
        components.append({"name": "P/FCF", "multiple": hist_pfcf, "fv": fv2})

    if hist_eveb and hist_eveb > 0 and ebitda and ebitda > 0 and shares and shares > 0:
        eq   = ebitda * hist_eveb - nd
        fv3  = round(eq / shares, 2) if eq > 0 else 0.0
        if fv3 > 0:
            components.append({"name": "EV/EBITDA", "multiple": hist_eveb, "fv": fv3})

    if not components:
        return None

    fair_value = round(sum(c["fv"] for c in components) / len(components), 2)

    return {
        "method":        "Mean Reversion",
        "fair_value":    fair_value,
        "mos_value":     round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "components":    components,
        "n_components":  len(components),
        "hist_pe_5y":    hist_pe,
        "hist_pfcf_5y":  hist_pfcf,
        "hist_eveb_5y":  hist_eveb,
        "reliable":      len(components) >= 2,
        "warning":       "Only 1 historical multiple available — estimate less reliable." if len(components) == 1 else None,
    }


# ── § 11  META MODELS ─────────────────────────────────────────────────────────

def run_bayesian_ensemble(
    d: dict,
    all_model_results: list,
    applicability_scores: dict,
    earnings_surprise_pct: float = None,
) -> dict:
    """
    Bayesian Ensemble — weighted consensus of all available model fair values.

    Weight for each model = its applicability_score (0-100).
    Fair value = sum(score_i × fv_i) / sum(score_i)

    Applies a small earnings surprise adjustment:
        +5% if last quarter beat estimates by >10%
        -5% if last quarter missed estimates by >10%

    Must be called AFTER all other models complete and applicability_scores
    are computed — it uses both as inputs.

    Returns None if fewer than 3 models have valid (> 0) fair values.
    """
    price    = d.get("price") or 0.0
    weighted = []

    for r in all_model_results:
        if not isinstance(r, dict):
            continue
        method = r.get("method", "")
        fv     = r.get("fair_value")
        score  = applicability_scores.get(method, 0)
        if fv and fv > 0 and score > 0:
            weighted.append((method, score, fv))

    if len(weighted) < 3:
        return None

    total_w    = sum(s for _, s, _ in weighted)
    raw_fv     = sum(s * fv for _, s, fv in weighted) / total_w

    # Earnings surprise adjustment
    surprise_adj = 0.0
    if earnings_surprise_pct is not None:
        if   earnings_surprise_pct >  10: surprise_adj =  0.05
        elif earnings_surprise_pct < -10: surprise_adj = -0.05

    fair_value = round(raw_fv * (1 + surprise_adj), 2)

    top3 = sorted(weighted, key=lambda x: x[1] * x[2], reverse=True)[:3]

    return {
        "method":               "Bayesian Ensemble",
        "fair_value":           fair_value,
        "mos_value":            round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "n_models_used":        len(weighted),
        "top_weighted_methods": [{"method": m, "score": s, "fv": fv} for m, s, fv in top3],
        "earnings_surprise_pct": earnings_surprise_pct,
        "surprise_adj_applied":  surprise_adj != 0.0,
        "reliable":              len(weighted) >= 5,
        "warning":               "Fewer than 5 models contributed — ensemble less robust." if len(weighted) < 5 else None,
    }


def run_multifactor_price_target(d: dict, benchmarks: dict) -> dict:
    """
    Multi-Factor Price Target.

    Scores 4 factors (0-25 pts each) and derives an expected annual return:
        expected_return = (total_score - 50) / 100 × 0.30 + 0.10
        Clipped to [7%, 40%]
        fair_value = price × (1 + expected_return)   (1-year target)

    Factors:
        Value    (0-25): PEG, P/FCF, EV/EBITDA relative to sector medians
        Quality  (0-25): ROIC, gross margin, FCF margin, operating margin
        Growth   (0-25): Revenue growth, EPS growth, forward growth estimate
        Momentum (0-25): TradingView rating, price performance, positioning

    Returns None if price is missing.
    """
    price = d.get("price")
    if not (price and price > 0):
        return None

    ext = d.get("ext") or {}

    # ── Factor 1: Value (0-25) ────────────────────────────────────────────────
    v_score = 0.0
    mkt_pe   = benchmarks.get("pe", 20.0)
    mkt_pfcf = benchmarks.get("pfcf", 18.0)
    mkt_eveb = benchmarks.get("ev_ebitda", 14.0)

    peg = d.get("peg")
    if peg and peg > 0:
        if   peg <= 0.75: v_score += 8
        elif peg <= 1.0:  v_score += 6
        elif peg <= 1.5:  v_score += 4
        elif peg <= 2.5:  v_score += 2

    curr_pfcf = d.get("current_pfcf")
    if curr_pfcf and curr_pfcf > 0:
        ratio = curr_pfcf / mkt_pfcf
        if   ratio <= 0.6: v_score += 7
        elif ratio <= 0.8: v_score += 5
        elif ratio <= 1.0: v_score += 3
        elif ratio <= 1.3: v_score += 1

    curr_eveb = d.get("current_ev_ebitda")
    if curr_eveb and curr_eveb > 0:
        ratio = curr_eveb / mkt_eveb
        if   ratio <= 0.6: v_score += 6
        elif ratio <= 0.8: v_score += 4
        elif ratio <= 1.0: v_score += 2
        elif ratio <= 1.3: v_score += 1

    # ── Factor 2: Quality (0-25) ──────────────────────────────────────────────
    q_score = 0.0
    roic = ext.get("roic")
    if roic is not None:
        if   roic >= 0.40: q_score += 10
        elif roic >= 0.25: q_score += 8
        elif roic >= 0.15: q_score += 5
        elif roic >= 0.08: q_score += 3
        elif roic >  0:    q_score += 1

    gm = d.get("gross_margin")
    if gm is not None:
        if   gm >= 70: q_score += 7
        elif gm >= 50: q_score += 5
        elif gm >= 35: q_score += 3
        elif gm >= 20: q_score += 1

    fcf_m = d.get("fcf_margin")
    if fcf_m is not None:
        if   fcf_m >= 20: q_score += 5
        elif fcf_m >= 10: q_score += 3
        elif fcf_m >= 5:  q_score += 1

    # ── Factor 3: Growth (0-25) ───────────────────────────────────────────────
    g_score   = 0.0
    rg = d.get("rev_growth_pct")
    if rg is not None:
        if   rg >= 40: g_score += 9
        elif rg >= 20: g_score += 7
        elif rg >= 10: g_score += 5
        elif rg >= 5:  g_score += 2

    eg = d.get("eps_growth_pct")
    if eg is not None and eg > 0:
        if   eg >= 40: g_score += 8
        elif eg >= 20: g_score += 6
        elif eg >= 10: g_score += 4
        elif eg >= 5:  g_score += 2

    fwd_g = d.get("est_growth")
    if fwd_g is not None:
        if   fwd_g >= 0.30: g_score += 5
        elif fwd_g >= 0.15: g_score += 3
        elif fwd_g >= 0.05: g_score += 1

    # ── Factor 4: Momentum (0-25) ─────────────────────────────────────────────
    m_score  = 0.0
    tv_rat   = d.get("tv_rating")
    if tv_rat is not None:
        pts = (tv_rat + 1) / 2 * 10    # map [-1,+1] → [0,10]
        m_score += min(10, pts)

    p3m = d.get("perf_3m")
    if p3m is not None:
        if   p3m >= 20: m_score += 8
        elif p3m >= 10: m_score += 6
        elif p3m >= 0:  m_score += 3
        elif p3m >= -10: m_score += 1

    pos52 = d.get("pos52")
    if pos52 is not None:
        if   pos52 >= 0.80: m_score += 5
        elif pos52 >= 0.60: m_score += 4
        elif pos52 >= 0.40: m_score += 3
        elif pos52 >= 0.20: m_score += 2

    # ── Totals ────────────────────────────────────────────────────────────────
    v_score = min(25.0, v_score)
    q_score = min(25.0, q_score)
    g_score = min(25.0, g_score)
    m_score = min(25.0, m_score)

    n_factors = sum(1 for s in [v_score, q_score, g_score, m_score] if s > 0)

    total_score      = v_score + q_score + g_score + m_score
    expected_return  = round(max(0.07, min(0.40, (total_score - 50) / 100 * 0.30 + 0.10)), 4)
    fair_value       = round(price * (1 + expected_return), 2)

    return {
        "method":           "Multi-Factor",
        "fair_value":       fair_value,
        "mos_value":        round(fair_value * (1 - MARGIN_OF_SAFETY), 2),
        "total_score":      round(total_score, 1),
        "value_score":      round(v_score, 1),
        "quality_score":    round(q_score, 1),
        "growth_score":     round(g_score, 1),
        "momentum_score":   round(m_score, 1),
        "expected_return":  round(expected_return * 100, 1),
        "n_factors":        n_factors,
        "reliable":         n_factors >= 3,
        "warning":          "Fewer than 3 factors available — score less reliable." if n_factors < 3 else None,
    }


# ── § 12  APPLICABILITY SCORING ───────────────────────────────────────────────

# Base precision scores from finance literature (out of 40)
_BASE_PRECISION = {
    "Monte Carlo DCF":     40,
    "Three-Stage DCF":     38,
    "ROIC Excess Return":  36,
    "RIM":                 35,
    "ERG":                 34,
    "DCF":                 33,
    "DDM":                 33,
    "FCF Yield":           31,
    "EV/NTM Rev":          30,
    "NCAV":                30,
    "EV/EBITDA":           30,
    "P/FCF":               29,
    "S-Curve TAM":         29,
    "P/E":                 28,
    "Fwd PEG":             28,
    "PIE":                 27,
    "Mean Reversion":      27,
    "Rule of 40":          26,
    "Bayesian Ensemble":   32,
    "Multi-Factor":        25,
    "TAM Scenario":        25,
    "Graham Number":       20,
    "Reverse DCF":          0,   # diagnostic only — exclude from fair-value ranking
}

# Key inputs required by each model for data-quality scoring
_KEY_INPUTS = {
    "DCF":             ["fcf", "shares", "est_growth"],
    "Three-Stage DCF": ["fcf", "shares", "est_growth"],
    "Monte Carlo DCF": ["fcf", "shares", "est_growth"],
    "P/FCF":           ["fcf_per_share", "est_growth"],
    "P/E":             ["eps", "est_growth"],
    "EV/EBITDA":       ["ebitda", "shares", "total_debt", "cash"],
    "FCF Yield":       ["fcf_per_share"],
    "RIM":             ["eps", "est_growth"],   # book_value_ps checked separately via ext
    "ROIC Excess Return": ["total_debt", "cash", "shares"],  # roic/equity checked via ext
    "DDM":             ["est_growth", "beta"],   # dividends_per_share checked via ext
    "NCAV":            ["shares"],               # balance sheet items checked via ext
    "Graham Number":   ["eps"],
    "Fwd PEG":         ["eps", "est_growth"],
    "EV/NTM Rev":      ["revenue", "est_growth", "shares"],
    "TAM Scenario":    ["revenue", "est_growth", "shares"],
    "S-Curve TAM":     ["revenue", "est_growth", "shares"],
    "Rule of 40":      ["revenue", "est_growth"],
    "ERG":             ["revenue", "est_growth", "shares"],
    "PIE":             ["price", "revenue", "shares"],
    "Mean Reversion":  ["eps"],
    "Bayesian Ensemble": ["price"],
    "Multi-Factor":    ["price"],
}


def score_model_applicability(
    model_name: str,
    result: dict,
    d: dict,
    reliability_flags: list,
) -> float:
    """
    Returns applicability score 0-100 for a model/company combination.

    Three additive components:
        base_precision  (0-40): academic/literature accuracy for this model type
        data_quality    (0-30): proportion of model's key inputs present and valid
        company_fit     (0-30): how well the model suits this specific company
    Deductions: -10 per reliability flag (min 0).

    This score is used to select the top-8 models to display prominently.
    """
    base = _BASE_PRECISION.get(model_name, 20)
    if base == 0:
        return 0.0   # diagnostic models (Reverse DCF) excluded

    # ── Data quality ──────────────────────────────────────────────────────────
    keys       = _KEY_INPUTS.get(model_name, [])
    n_present  = sum(1 for k in keys if d.get(k) not in (None, 0, ""))
    data_q     = round((n_present / max(len(keys), 1)) * 30, 1) if keys else 15.0

    # Extra ext checks for models that need d["ext"]
    ext = d.get("ext") or {}
    if model_name == "RIM":
        if ext.get("book_value_ps", 0) > 0:   data_q = min(30, data_q + 5)
    if model_name == "ROIC Excess Return":
        if ext.get("roic") is not None:        data_q = min(30, data_q + 8)
        if ext.get("stockholders_equity"):     data_q = min(30, data_q + 4)
    if model_name == "DDM":
        if (ext.get("dividends_per_share") or 0) > 0: data_q = min(30, data_q + 15)
        else: data_q = max(0, data_q - 20)    # big penalty — DDM useless without divs
    if model_name == "NCAV":
        if ext.get("total_current_assets") and ext.get("total_liabilities"):
            data_q = min(30, data_q + 12)
        else:
            data_q = 0.0   # cannot run without balance sheet
    if model_name == "Mean Reversion":
        n_hist = sum(1 for k in ["hist_pe_5y","hist_pfcf_5y","hist_eveb_5y"] if ext.get(k))
        data_q = round(n_hist / 3 * 30, 1)

    # ── Company fit ───────────────────────────────────────────────────────────
    fit    = 15.0   # neutral baseline
    fcf    = d.get("fcf") or 0
    growth = d.get("est_growth") or 0
    sector = d.get("sector") or ""
    gm     = d.get("gross_margin") or 0
    divs   = (ext.get("dividends_per_share") or 0)
    roic   = ext.get("roic")
    bvps   = ext.get("book_value_ps") or 0

    # FCF quality boosts for cash-flow models
    if model_name in ("DCF", "Three-Stage DCF", "Monte Carlo DCF", "FCF Yield"):
        if fcf > 0: fit += 6
        else:       fit -= 8

    # Growth rate fit
    if model_name in ("ERG", "EV/NTM Rev", "S-Curve TAM", "Fwd PEG", "TAM Scenario"):
        if growth > 0.25: fit += 8
        elif growth > 0.15: fit += 4
        elif growth < 0.05: fit -= 6

    if model_name in ("P/E", "EV/EBITDA", "Mean Reversion", "ROIC Excess Return"):
        if growth < 0.10: fit += 6
        elif growth > 0.30: fit -= 4

    # Sector fit
    if sector in ("Financials", "Real Estate"):
        if model_name == "RIM":     fit += 8
        if model_name in ("DCF", "Three-Stage DCF", "Monte Carlo DCF"): fit -= 6

    # Dividend fit
    if model_name == "DDM":
        if divs > 0: fit += 15
        else:        fit -= 20

    # Balance sheet models
    if model_name in ("RIM", "NCAV"):
        if bvps > 0: fit += 6

    # Gross margin fit
    if model_name in ("Rule of 40", "ERG"):
        if gm > 60: fit += 4

    # ROIC model — needs ROIC data and rewards high ROIC
    if model_name == "ROIC Excess Return":
        if roic is not None and roic > 0: fit += 8

    # S-Curve TAM best for hypergrowth, pre-maturity companies
    if model_name == "S-Curve TAM":
        if growth > 0.30: fit += 6
        elif growth < 0.10: fit -= 8

    # PIE — more useful when market and analyst expectations diverge
    if model_name == "PIE":
        pass   # neutral — applicable to any company with revenue data

    fit = max(0.0, min(30.0, fit))

    # ── Reliability penalty ───────────────────────────────────────────────────
    penalty = len(reliability_flags) * 10

    score = base + data_q + fit - penalty
    return round(max(0.0, min(100.0, score)), 1)
