import sys
import os
import argparse
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
forwardDCF.py: Intrinsic Valuation & Price Target Forecasting Tool

This script performs a 2-Stage Discounted Cash Flow (DCF) analysis to determine
a stock's current intrinsic value and projects a 12-month forward price target.

WHAT IT DOES:
    1.  Calculates WACC: Estimates the Weighted Average Cost of Capital using
        CAPM and tax-shielded interest rate for cost of debt.
    2.  10-Year 2-Stage DCF: Decay model for years 1-5, linear fade years 6-10.
    3.  Forward Target: 'Rolls forward' fair value one year at the Cost of Equity.
    4.  Market Analysis: Side-by-side of intrinsic value, 1-year target, and price.

HOW TO USE IT:
    python forwardDCF.py NVDA --growth 0.15

FORMULAS:
    - Current Fair Value = library run_dcf() (10-year 2-stage model)
    - One-Year Target Price = Fair Value * (1 + Cost of Equity)
    - Expected 12M Return = (One-Year Target / Current Price) - 1
"""

from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, calculate_wacc, run_dcf


def calculate_dcf(ticker_symbol, growth_rate=0.10):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # 1. Robust FCF Retrieval
        cash_flow = ticker.cashflow
        if 'Free Cash Flow' in cash_flow.index:
            current_fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
        else:
            current_fcf = info.get('freeCashflow')

        if current_fcf is None:
            print(f"Error: Could not find FCF data for {ticker_symbol}.")
            return

        # 2. Fetch yfinance WACC inputs
        financials = ticker.financials
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

        beta = info.get('beta', 1.0)

        # 3. Build canonical data dict
        d = {
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

        # 4. Run DCF via library
        result = run_dcf(d)
        if result["fair_value"] is None:
            print(f"DCF could not compute a fair value for {ticker_symbol}.")
            return

        fair_value = result["fair_value"]
        wacc       = result["wacc"]
        ke         = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

        # 5. One-Year Target
        one_year_target = fair_value * (1 + ke)
        curr_price      = info.get('currentPrice')
        high_52         = info.get('fiftyTwoWeekHigh', 0)
        low_52          = info.get('fiftyTwoWeekLow', 0)

        print(f"\n--- DCF VALUATION & PRICE TARGET: {ticker_symbol.upper()} ---")
        print(f"Discount Rate (WACC): {wacc:.2%}")
        print(f"Cost of Equity (Ke):  {ke:.2%}")
        print(f"Growth Assumption:    {growth_rate:.1%}")
        print("-" * 50)
        print(f"CURRENT FAIR VALUE:   ${fair_value:.2f}")
        print(f"ONE-YEAR TARGET:      ${one_year_target:.2f}")
        print(f"CURRENT MARKET PRICE: ${curr_price:.2f}")
        total_upside = (one_year_target / curr_price) - 1
        print(f"EXPECTED 12M RETURN:  {total_upside:.1%}")
        print("-" * 50)
        print(f"52-WEEK RANGE:        ${low_52:.2f} - ${high_52:.2f}")

        if fair_value > high_52:
            print("Note: Fair value exceeds 52-week high. Check growth assumptions.")
        elif fair_value < low_52:
            print("Note: Fair value is below 52-week low. Asset may be overvalued.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCF Valuation with One-Year Target Price")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--growth", type=float, default=0.10,
                        help="5-year FCF growth rate")
    args = parser.parse_args()
    calculate_dcf(args.ticker, growth_rate=args.growth)
