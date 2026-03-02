import sys
import os
import argparse
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
WACCcalc.py: Professional Weighted Average Cost of Capital (WACC) Estimator

This script automates the calculation of WACC, a critical discount rate used
to determine the present value of a company's future cash flows. It is the
foundation for Discounted Cash Flow (DCF) models.

WHAT IT DOES:
    1.  Calculates Cost of Equity (Ke): Uses the Capital Asset Pricing Model
        (CAPM) with a 10Y Treasury proxy (4.3%) and a standard Equity Risk
        Premium (5.5%).
    2.  Calculates After-Tax Cost of Debt (Kd): Fetches interest expenses and
        total debt. It derives an effective tax rate by comparing tax
        provisions to pretax income.
    3.  Weighted Average: Dynamically weights the components based on the
        company's current market capital structure (Market Cap vs. Debt).

HOW TO USE IT:
    Run the script from the terminal with a ticker symbol. You can also
    override the Beta manually to test different risk scenarios.

    Example (Standard run):
        python WACCcalc.py AAPL

    Example (Custom Beta override):
        python WACCcalc.py TSLA --beta 1.4

FORMULAS:
    - Ke = Risk-Free Rate + (Beta * Equity Risk Premium)
    - Kd (After-Tax) = (Interest Expense / Total Debt) * (1 - Tax Rate)
    - WACC = (Equity_Weight * Ke) + (Debt_Weight * Kd)
"""

from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, calculate_wacc


def run_wacc_analysis(ticker_symbol, custom_beta=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        if 'marketCap' not in info:
            print(f"Error: Could not find market data for {ticker_symbol}.")
            return

        # Fetch TTM interest and tax data
        income_stmt = ticker.financials
        interest_expense = 0
        for label in ['Interest Expense', 'Interest Expense Non Operating']:
            if label in income_stmt.index:
                val = income_stmt.loc[label].iloc[0]
                interest_expense = abs(val) if not pd.isna(val) else 0
                if interest_expense > 0:
                    break

        income_tax_expense, pretax_income = None, None
        try:
            income_tax_expense = income_stmt.loc['Tax Provision'].iloc[0]
            pretax_income      = income_stmt.loc['Pretax Income'].iloc[0]
        except Exception:
            pass

        # Build canonical data dict
        beta = custom_beta if custom_beta is not None else info.get('beta', 1.0)
        d = {
            "market_cap":  info.get('marketCap', 0),
            "total_debt":  info.get('totalDebt', 0),
            "beta":        beta,
            "wacc_raw": {
                "beta_yf":            beta,
                "interest_expense":   interest_expense,
                "income_tax_expense": income_tax_expense,
                "pretax_income":      pretax_income,
                "total_debt_yf":      info.get('totalDebt', 0),
            },
        }

        wacc, source = calculate_wacc(d)
        ke = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

        print(f"\n--- WACC Analysis: {ticker_symbol.upper()} ---")
        print(f"Beta: {beta:.2f}")
        print(f"Cost of Equity (Ke): {ke:.2%}")
        print("-" * 35)
        print(f"ESTIMATED WACC: {wacc:.2%}")
        print(f"Source: {source}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--beta", type=float, help="Manual Beta override")
    args = parser.parse_args()
    run_wacc_analysis(args.ticker, custom_beta=args.beta)
