import sys
import os
import argparse
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
predictiveERG.py: Expert Multi-Metric Growth Stock Valuation Dashboard

This script provides a high-level valuation summary by bridging two different
financial worlds: Revenue-based multiples (ERG) and Cash-Flow-based
fundamentals (DCF).

WHAT IT DOES:
    1.  Calculates ERG Fair Value: Top-line valuation normalizing EV/Revenue
        against revenue growth. Ideal for high-growth companies.
    2.  Performs Reverse DCF: Calculates exactly what annual growth rate is
        "priced in" by the current market using Brent's root-finding method.
    3.  Growth Gap Analysis: Compares expert growth vs market-implied growth.
    4.  Manual Override: Expert Mode for custom growth theses.

HOW TO USE IT:
    python predictiveERG.py TSLA --growth 0.25
    python predictiveERG.py NVDA --erg 0.65

FORMULAS:
    - ERG Fair Value = ((Target ERG * Growth * NTM Revenue) - Net Debt) / Shares
    - Market Implied Growth = Root(DCF(g) - Current Price = 0)
"""

from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, TERMINAL_GROWTH_RATE, calculate_wacc, run_dcf


def run_expert_analysis(ticker_symbol, manual_g=None, target_erg=0.5):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info   = ticker.info
        current_price = info.get('currentPrice')
        shares        = info.get('sharesOutstanding')
        net_debt      = info.get('totalDebt', 0) - info.get('totalCash', 0)

        # 1. SET GROWTH RATE
        if manual_g:
            g_rate = manual_g
            source = "Manual Expert Override"
        else:
            g_rate = info.get('revenueGrowth', 0.10)
            source = "System Default"

        # 2. ERG VALUATION
        ntm_rev = info.get('revenueEstimateNextYear') or (info.get('totalRevenue', 0) * (1 + g_rate))
        fair_ev = (target_erg * (g_rate * 100)) * ntm_rev
        erg_fair_value = (fair_ev - net_debt) / shares

        # 3. REVERSE DCF — build canonical dict and use library WACC
        fcf = info.get('freeCashflow')
        if fcf is None:
            cash_stmt = ticker.cashflow
            fcf = cash_stmt.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_stmt.index else 0

        financials = ticker.financials
        interest_expense = 0
        try:
            val = abs(financials.loc['Interest Expense'].iloc[0])
            if not pd.isna(val):
                interest_expense = val
        except Exception:
            pass

        beta = info.get('beta', 1.2)
        d = {
            "fcf":        fcf,
            "shares":     shares,
            "total_debt": info.get('totalDebt', 0),
            "cash":       info.get('totalCash', 0),
            "est_growth": g_rate,
            "market_cap": info.get('marketCap', 0),
            "beta":       beta,
            "price":      current_price,
            "wacc_raw": {
                "beta_yf":          beta,
                "interest_expense": interest_expense,
                "total_debt_yf":    info.get('totalDebt', 0),
            },
        }
        wacc, _ = calculate_wacc(d)
        wacc = max(wacc, 0.07)  # floor at 7% (conservative)

        # Solve for implied growth using flat 5-year DCF (for solver simplicity)
        try:
            def flat_dcf(g):
                pv_s1 = sum([(fcf * (1 + g)**t) / (1 + wacc)**t for t in range(1, 6)])
                fcf5  = fcf * (1 + g)**5
                tv    = (fcf5 * (1 + TERMINAL_GROWTH_RATE)) / (wacc - TERMINAL_GROWTH_RATE)
                pv_tv = tv / (1 + wacc)**5
                return (pv_s1 + pv_tv - net_debt) / shares - current_price
            implied_g = brentq(flat_dcf, -0.2, 1.5)
        except Exception:
            implied_g = 0.0

        # OUTPUT DASHBOARD
        print(f"\n{'='*55}")
        print(f" EXPERT VALUATION DASHBOARD: {ticker_symbol.upper()} ")
        print(f"{'='*55}")
        print(f"Current Price:      ${current_price:.2f}")
        print(f"Growth Basis:       {g_rate:.1%} ({source})")
        print(f"Market Implied G:   {implied_g:.1%} (Reverse DCF)")
        print(f"{'-'*55}")
        print(f"ERG FAIR VALUE:     ${erg_fair_value:.2f}")
        gap = g_rate - implied_g
        print(f"GROWTH GAP:         {gap:+.1%} (vs Market expectations)")
        print(f"{'-'*55}")

        if g_rate > implied_g:
            print("VERDICT: UNDERVALUED. You are more bullish than the market.")
        else:
            print("VERDICT: OVERVALUED. The market expects more than your forecast.")
        print(f"{'='*55}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--growth", type=float, help="Manual Growth Rate (e.g. 0.45)")
    parser.add_argument("--erg", type=float, default=0.5, help="Target ERG Multiple")
    args = parser.parse_args()
    run_expert_analysis(args.ticker, args.growth, args.erg)
