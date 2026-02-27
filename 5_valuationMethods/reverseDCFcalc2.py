import sys
import os
import argparse
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Library equivalents: valuation_models.run_erg_valuation(d), valuation_models.run_reverse_dcf(d)
from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, MARGIN_OF_SAFETY

"""
reverseDCFcalc2.py: Robust Revenue-Based (ERG) Valuation Tool

This script implements the ERG (EV/Revenue/Growth) methodology to determine the 
intrinsic value of a stock. It is specifically designed for growth-oriented 
companies where top-line expansion is the primary driver of value.

WHAT IT DOES:
    1.  Determines NTM Revenue: Prioritizes analyst consensus estimates for next 
        year's revenue. If unavailable, it calculates a 3-year historical CAGR 
        to project forward revenue.
    2.  Calculates ERG Multiples: Normalizes the company's current Enterprise 
        Value (EV) against its revenue growth to see how much the market is 
        paying for each unit of growth.
    3.  Intrinsic Value: Derives a fair share price based on a user-defined 
        Target ERG (default is 0.5, implying a fair EV/Rev multiple is half 
        the growth rate).
    4.  Liquidity-Adjusted Net Debt: Subtracts both cash and short-term 
        investments from total debt to get a cleaner view of the company's 
        enterprise value.
    5.  Margin of Safety (MOS): Provides a "Buy Price" target by applying a 
        discount (default 20%) to the calculated intrinsic value.

HOW TO USE IT:
    Run the script from the command line with a ticker symbol. You can 
    optionally adjust the Target ERG or the Margin of Safety.

    Example (Standard run):
        python reverseDCFcalc2.py TSLA

    Example (Conservative run with 30% Margin of Safety):
        python reverseDCFcalc2.py SNOW --erg 0.4 --mos 0.3

FORMULAS:
    - CAGR = (Recent Revenue / Old Revenue)^(1/n) - 1
    - Fair EV Multiple = Target ERG * (Revenue Growth Rate * 100)
    - Intrinsic Price = ((Fair EV Multiple * NTM Revenue) - Net Debt) / Shares
"""

def get_ntm_revenue_robust(ticker_obj):
    """
    Robustly determines NTM Revenue.
    Prioritizes analyst estimates, falls back to an accurate 3-year CAGR.
    """
    info = ticker_obj.info
    
    # 1. Primary: Analyst Consensus
    ntm_revenue = info.get('revenueEstimateNextYear')
    if ntm_revenue:
        current_rev = info.get('totalRevenue', 1)
        implied_growth = (ntm_revenue / current_rev) - 1
        return ntm_revenue, implied_growth, "Analyst Consensus"

    # 2. Fallback: Calculate 3-Year Historical CAGR
    try:
        rev_history = ticker_obj.financials.loc['Total Revenue']
        if len(rev_history) >= 4: # Need 4 points for 3 full years of growth
            recent_rev = rev_history.iloc[0]
            old_rev = rev_history.iloc[3]
            # Correct CAGR for 3 intervals: (Ending / Beginning)^(1/3) - 1
            cagr = (recent_rev / old_rev) ** (1/3) - 1
            source = "3-Year Hist. CAGR"
        else:
            cagr = info.get('revenueGrowth', 0.10) 
            source = "Quarterly Snapshot/Default"
            
        current_rev = info.get('totalRevenue', 1)
        # Apply a 1% floor to avoid division by zero in ERG calculation
        cagr = max(cagr, 0.01)
        ntm_revenue = current_rev * (1 + cagr)
        return ntm_revenue, cagr, source

    except Exception:
        return info.get('totalRevenue', 1) * 1.1, 0.10, "Default 10% Projection"

def calculate_erg_valuation(ticker_symbol, target_erg=0.5, mos=0.20):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. Robust Revenue & Growth Data
        ntm_revenue, growth_rate, source = get_ntm_revenue_robust(ticker)
        growth_pts = growth_rate * 100
        
        # 2. Market & Liquidity Data (Including Short Term Investments)
        current_price = info.get('currentPrice')
        shares = info.get('sharesOutstanding')
        cash_and_equiv = info.get('totalCash', 0)
        # Net Debt = Total Debt - (Cash + Short Term Investments)
        net_debt = info.get('totalDebt', 0) - cash_and_equiv
        
        # 3. Current ERG Calculation
        current_ev = (current_price * shares) + net_debt
        current_rev_multiple = current_ev / ntm_revenue
        current_erg = current_rev_multiple / growth_pts

        # 4. Fair Value Calculation
        fair_multiple = target_erg * growth_pts
        fair_ev = fair_multiple * ntm_revenue
        intrinsic_value = (fair_ev - net_debt) / shares
        
        # 5. Margin of Safety (MOS)
        mos_price = intrinsic_value * (1 - mos)

        print(f"\n--- ERG VALUATION (ROBUST): {ticker_symbol.upper()} ---")
        print(f"Current Price:       ${current_price:.2f}")
        print(f"Growth Basis:        {growth_pts:.2f}% ({source})")
        print(f"NTM Revenue Est:     ${ntm_revenue/1e9:.2f}B")
        print("-" * 55)
        print(f"Current EV/Rev:      {current_rev_multiple:.2f}x")
        print(f"Current ERG Ratio:   {current_erg:.2f}")
        print("-" * 55)
        print(f"INTRINSIC VALUE:     ${intrinsic_value:.2f}")
        print(f"MOS BUY PRICE:       ${mos_price:.2f} ({int(mos*100)}% Discount)")
        print(f"POTENTIAL UPSIDE:    {((intrinsic_value/current_price)-1):.1%}")
        print("-" * 55)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--erg", type=float, default=0.5, help="Target ERG Ratio")
    parser.add_argument("--mos", type=float, default=0.20, help="Margin of Safety (decimal)")
    args = parser.parse_args()
    calculate_erg_valuation(args.ticker, args.erg, args.mos)