import sys
import os
import argparse
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Library equivalent for richer multi-scenario output: valuation_models.run_forward_peg(d)
from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM

"""
forwardPEG.py: Forward-Looking PEG Ratio Valuation Tool

This script calculates the intrinsic 'Fair Market Value' of a stock using the 
Price/Earnings-to-Growth (PEG) ratio. It is designed to value growth companies 
by normalizing their P/E ratio against their expected earnings growth.

WHAT IT DOES:
    1.  Earnings Data: Fetches Analyst Consensus Forward EPS (Next Twelve Months). 
        Falls back to Trailing Twelve Months (TTM) EPS if estimates are missing.
    2.  Growth Assessment: Prioritizes projected 5-year annualized growth rates. 
        If data is unavailable, it applies a conservative default to ensure 
        the model produces a baseline valuation.
    3.  Live PEG Analysis: Calculates the company's current PEG based on today's 
        market price and forward earnings.
    4.  Fair Value Calculation: Derives a target share price based on a user-defined 
        PEG benchmark (Default is 1.0, signifying growth is perfectly priced).
    5.  Opportunity Analysis: Compares the fair value to the current market price 
        to determine the percentage upside or downside.

HOW TO USE IT:
    Run the script from the command line with a ticker symbol and an optional 
    target PEG ratio.

    Example (Standard valuation for Apple):
        python forwardPEG.py AAPL --peg 1.0

    Example (Aggressive valuation for a high-growth AI stock):
        python forwardPEG.py NVDA --peg 1.5

FORMULAS:
    - Forward P/E = Current Price / Forward EPS
    - Current PEG = Forward P/E / (Growth Rate * 100)
    - Fair Price = Target PEG * Forward EPS * (Growth Rate * 100)
"""


def calculate_peg_valuation(ticker_symbol, target_peg=1.0):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. Get Forward Earnings (NTM EPS)
        forward_eps = info.get('forwardEps')
        if not forward_eps:
            forward_eps = info.get('trailingEps', 0)
            eps_source = "Trailing EPS (Forward unavailable)"
        else:
            eps_source = "Forward EPS (Analyst Consensus)"

        if not forward_eps or forward_eps <= 0:
            print(f"Error: No positive earnings data available for {ticker_symbol}.")
            return

        # 2. Get Growth Rate (Prioritizing Projected over Historical)
        growth_rate = info.get('longTermProxyGrowth') 
        growth_source = "Analyst 5Y Projection"
        
        if not growth_rate:
            growth_rate = info.get('earningsQuarterlyGrowth')
            growth_source = "Quarterly Earnings Growth"
        
        if not growth_rate or growth_rate <= 0:
            growth_rate = 0.15 
            growth_source = "Conservative Default (15%)"

        growth_value_for_peg = growth_rate * 100 # PEG uses whole numbers

        # 3. Market Data
        current_price = info.get('currentPrice')
        if not current_price:
            print("Error: Could not retrieve current market price.")
            return

        current_forward_pe = current_price / forward_eps
        current_peg = current_forward_pe / growth_value_for_peg

        # 4. Fair Value Calculation
        fair_price = target_peg * forward_eps * growth_value_for_peg

        # 5. Output Results
        print(f"\n--- FORWARD PEG VALUATION: {ticker_symbol.upper()} ---")
        print(f"Current Price:       ${current_price:.2f}")
        print(f"Forward EPS:         ${forward_eps:.2f} ({eps_source})")
        print(f"Est. Growth Rate:    {growth_value_for_peg:.2f}% ({growth_source})")
        print("-" * 50)
        print(f"Current Forward P/E: {current_forward_pe:.2f}x")
        print(f"Current PEG Ratio:   {current_peg:.2f}")
        print("-" * 50)
        print(f"TARGET PEG:          {target_peg:.2f}")
        print(f"FAIR MARKET VALUE:   ${fair_price:.2f}")
        
        upside = (fair_price / current_price) - 1
        print(f"POTENTIAL UPSIDE:    {upside:.1%}")
        print("-" * 50)

    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--peg", type=float, default=1.0, help="Target PEG ratio (default 1.0)")
    args = parser.parse_args()
    
    calculate_peg_valuation(args.ticker, args.peg)