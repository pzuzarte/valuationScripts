import sys
import os
import argparse
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
fullDCF.py: Professional Scenario & Reverse DCF Analysis Tool

This script provides an intrinsic valuation of a company using a 2-Stage
Discounted Cash Flow (DCF) model. It evaluates a stock under multiple
growth scenarios (Base, Bull, Bear) and calculates the "market implied"
growth rate via a Reverse DCF.

WHAT IT DOES:
    1.  Calculates WACC: Dynamically estimates WACC via library calculate_wacc().
    2.  Manual Beta Override: Normalizes WACC against extreme market volatility.
    3.  Robust FCF Retrieval: Attempts to pull reported FCF; falls back to OCF-CapEx.
    4.  Scenario Analysis: Projects FCF under Bear/Base/Bull growth assumptions.
    5.  Reverse DCF: Brent's method to find market-implied FCF growth rate.
    6.  One-Year Targets: Compounds fair value by Cost of Equity (Ke).

HOW TO USE IT:
    python fullDCF.py MSFT --growth 0.15
    python fullDCF.py MSFT --growth 0.10 --beta 1.0 --spread 0.03

FORMULAS:
    - Ke = Risk Free Rate + (Beta * Equity Risk Premium)
    - WACC via library calculate_wacc()
    - Fair Value via library run_dcf() (10-year 2-stage with decay)
"""

from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, TERMINAL_GROWTH_RATE, calculate_wacc, run_dcf


def _build_d(info, financials, current_fcf, growth_rate, manual_beta=None):
    """Build canonical data dict from yfinance data."""
    beta = manual_beta if manual_beta is not None else info.get('beta', 1.0)

    interest_expense = 0
    for label in ['Interest Expense', 'Interest Expense Non Operating']:
        if label in financials.index:
            val = financials.loc[label].iloc[0]
            interest_expense = abs(val) if not pd.isna(val) else 0
            if interest_expense > 0:
                break

    income_tax_expense, pretax_income = None, None
    try:
        income_tax_expense = financials.loc['Tax Provision'].iloc[0]
        pretax_income      = financials.loc['Pretax Income'].iloc[0]
    except Exception:
        pass

    return {
        "fcf":        current_fcf,
        "shares":     info.get('sharesOutstanding'),
        "total_debt": info.get('totalDebt', 0),
        "cash":       info.get('totalCash', 0),
        "est_growth": growth_rate,
        "market_cap": info.get('marketCap', 0),
        "beta":       beta,
        "price":      info.get('currentPrice'),
        "wacc_raw": {
            "beta_yf":            beta,
            "interest_expense":   interest_expense,
            "income_tax_expense": income_tax_expense,
            "pretax_income":      pretax_income,
            "total_debt_yf":      info.get('totalDebt', 0),
        },
    }


def run_pro_analysis(ticker_symbol, base_growth, spread, manual_beta):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info   = ticker.info

        # 1. Robust FCF Retrieval
        cashflow = ticker.cashflow
        if 'Free Cash Flow' in cashflow.index:
            current_fcf = cashflow.loc['Free Cash Flow'].iloc[0]
        else:
            ocf   = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
            capex = abs(cashflow.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cashflow.index else 0
            current_fcf = ocf - capex

        if current_fcf <= 0:
            print(f"Warning: {ticker_symbol} has negative FCF. DCF results may be invalid.")

        financials    = ticker.financials
        current_price = info.get('currentPrice')
        analyst_target = info.get('targetMeanPrice', "N/A")

        # Use base-case d to get WACC and Ke
        d_base = _build_d(info, financials, current_fcf, base_growth, manual_beta)
        wacc, _ = calculate_wacc(d_base)
        beta_used = manual_beta if manual_beta is not None else info.get('beta', 1.0)
        ke = RISK_FREE_RATE + beta_used * EQUITY_RISK_PREMIUM

        scenarios = {
            "Bear Case": base_growth - spread,
            "Base Case": base_growth,
            "Bull Case": base_growth + spread,
        }

        print(f"\n--- PRO VALUATION: {ticker_symbol.upper()} ---")
        print(f"Current Price: ${current_price:.2f} | Analyst Mean Target: ${analyst_target}")
        print(f"WACC: {wacc:.2%} | Cost of Equity (Ke): {ke:.2%} | Beta: {beta_used}")
        print("-" * 75)
        print(f"{'Scenario':<15} | {'Growth':<10} | {'Fair Value':<12} | {'1Y Target':<12} | {'Upside':<8}")
        print("-" * 75)

        for name, g in scenarios.items():
            d = _build_d(info, financials, current_fcf, g, manual_beta)
            result = run_dcf(d)
            fv = result["fair_value"] if result["fair_value"] else 0.0
            target = fv * (1 + ke)
            upside = (target / current_price) - 1
            print(f"{name:<15} | {g:<10.1%} | ${fv:<11.2f} | ${target:<11.2f} | {upside:>8.1%}")

        # 2. Reverse DCF — find implied growth via simple DCF formula (flat 5-year for solver)
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        shares   = info.get('sharesOutstanding')

        def flat_dcf(g):
            pv_s1 = sum([(current_fcf * (1 + g)**yr) / (1 + wacc)**yr for yr in range(1, 6)])
            fcf5  = current_fcf * (1 + g)**5
            tv    = (fcf5 * (1 + TERMINAL_GROWTH_RATE)) / (wacc - TERMINAL_GROWTH_RATE)
            pv_tv = tv / (1 + wacc)**5
            return (pv_s1 + pv_tv - net_debt) / shares - current_price

        try:
            implied_g = brentq(flat_dcf, -0.9, 2.0)
            print("-" * 75)
            print(f"REVERSE DCF: The market implies {implied_g:.2%} annual FCF growth.")
        except Exception:
            print("-" * 75)
            print("REVERSE DCF: Market price too extreme for the model to find implied growth.")

    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--growth", type=float, default=0.15,
                        help="Base FCF growth rate (decimal)")
    parser.add_argument("--spread", type=float, default=0.05,
                        help="Growth variance for bull/bear (decimal)")
    parser.add_argument("--beta", type=float, help="Manual Beta override")
    args = parser.parse_args()
    run_pro_analysis(args.ticker, args.growth, args.spread, args.beta)
