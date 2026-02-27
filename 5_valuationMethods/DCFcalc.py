import sys
import os
import argparse
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
DCFcalc.py: Professional 2-Stage Discounted Cash Flow (DCF) Valuation Tool

This script determines the intrinsic 'Fair Value' of a stock by projecting
future Free Cash Flows (FCF) and discounting them back to the present value
using the Weighted Average Cost of Capital (WACC).

WHAT IT DOES:
    1.  Estimates/Overrides WACC: Dynamically estimates the discount rate
        using CAPM or accepts a manual override via command line.
    2.  Multi-Source FCF: Pulls FCF from ticker info or the cash flow statement,
        ensuring the script executes even if metadata is sparse.
    3.  10-Year 2-Stage Projection: Projects growth with a decay model for
        years 1-5, linear fade to terminal in years 6-10, then Gordon Growth.
    4.  Market Comparison: Compares Fair Value against the current price
        to identify potential upside or downside.

HOW TO USE IT:
    Run from the terminal with a ticker and optional growth or WACC values.

    Example (Manual 9% WACC and 15% growth):
        python DCFcalc.py MSFT --growth 0.15 --wacc 0.09

FORMULAS:
    - Ke = RiskFree + (Beta * EquityRiskPremium)
    - Terminal Value = (FCF_Yr10 * (1 + g_long)) / (WACC - g_long)
    - Fair Price = (PV_Stage1+2 + PV_Terminal - Net_Debt) / Shares
"""

from valuation_models import calculate_wacc, run_dcf


def calculate_dcf(ticker_symbol, growth_rate, manual_wacc=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # 1. Robust FCF Retrieval
        cashflow_stmt = ticker.cashflow
        current_fcf = info.get('freeCashflow')
        if current_fcf is None:
            if 'Free Cash Flow' in cashflow_stmt.index:
                current_fcf = cashflow_stmt.loc['Free Cash Flow'].iloc[0]
            else:
                ocf   = cashflow_stmt.loc['Operating Cash Flow'].iloc[0]
                capex = cashflow_stmt.loc['Capital Expenditure'].iloc[0]
                current_fcf = ocf + capex

        shares = info.get('sharesOutstanding')

        # 2. Fetch yfinance WACC inputs for library
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

        # 3. Build canonical data dict
        d = {
            "fcf":          current_fcf,
            "shares":       shares,
            "total_debt":   info.get('totalDebt', 0),
            "cash":         info.get('totalCash', 0),
            "est_growth":   growth_rate,
            "market_cap":   info.get('marketCap', 0),
            "beta":         info.get('beta', 1.0),
            "price":        info.get('currentPrice'),
            "wacc_override": manual_wacc,
            "wacc_raw": {
                "beta_yf":             info.get('beta', 1.0),
                "interest_expense":    interest_expense,
                "income_tax_expense":  income_tax_expense,
                "pretax_income":       pretax_income,
                "total_debt_yf":       info.get('totalDebt', 0),
            },
        }

        # 4. Run DCF via library (10-year 2-stage model)
        result = run_dcf(d)
        if result["fair_value"] is None:
            print(f"DCF could not compute a fair value for {ticker_symbol}.")
            return

        fair_value   = result["fair_value"]
        wacc         = result["wacc"]
        curr_price   = info.get('currentPrice')
        wacc_source  = "Manual Override" if manual_wacc else "Estimated (CAPM)"

        print(f"\n--- DCF VALUATION: {ticker_symbol.upper()} ---")
        print(f"WACC:         {wacc:.2%} ({wacc_source})")
        print(f"Growth Rate:  {growth_rate:.1%}")
        print("-" * 45)
        print(f"FAIR VALUE:   ${fair_value:.2f}")
        print(f"MARKET PRICE: ${curr_price:.2f}")
        print(f"UPSIDE:       {((fair_value / curr_price) - 1):.1%}")
        if result.get("warning"):
            print(f"WARNING:      {result['warning']}")
        print("-" * 45)

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--growth", type=float, default=0.10,
                        help="Stage 1 annual growth rate (e.g. 0.15)")
    parser.add_argument("--wacc", type=float,
                        help="Manual WACC override (e.g. 0.08)")
    args = parser.parse_args()
    calculate_dcf(args.ticker, args.growth, manual_wacc=args.wacc)
