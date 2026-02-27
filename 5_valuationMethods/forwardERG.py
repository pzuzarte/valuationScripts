import sys
import os
import argparse
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Library equivalent for full dual-lens ERG: valuation_models.run_erg_valuation(d)
from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM, MARGIN_OF_SAFETY

"""
forwardERG.py: Revenue Growth Valuation Tool (ERG Methodology)

This script implements the ERG (Enterprise-value-to-Revenue-to-Growth) methodology 
to value high-growth companies. This model is particularly useful for sectors 
where top-line expansion is the primary driver of value (SaaS, AI, Tech).

WHAT IT DOES:
    1.  Calculates Enterprise Value (EV): Combines market capitalization with net 
        debt (Total Debt - Total Cash) to reflect the true cost of the revenue stream.
    2.  Projects Forward Revenue: Prioritizes analyst consensus estimates for Next 
        Twelve Months (NTM) revenue. If unavailable, it calculates a 3-year historical 
        CAGR as a robust fallback.
    3.  Calculates Current ERG: Derives the "live" ERG multiple the market is 
        currently paying based on the current stock price.
    4.  Applies Target ERG: Allows the user to value the stock based on the current 
        market derived ERG or a manually defined target (e.g., 0.5x).
    5.  Implies Fair Price: Calculates the intrinsic share price based on the 
        selected ERG target.

HOW TO USE IT:
    Run the script from the terminal with a ticker symbol. Provide an optional 
    target ERG to see how it affects the valuation.

    Example 1 (See current market valuation):
        python forwardERG.py NVDA

    Example 2 (Value based on a conservative 0.45 ERG):
        python forwardERG.py NVDA --erg 0.45

FORMULAS:
    - Current ERG = (EV / NTM Revenue) / (Revenue Growth Rate * 100)
    - Fair EV Multiple = Target ERG * (Revenue Growth Rate * 100)
    - Fair Price = ((Fair EV Multiple * NTM Revenue) - Net Debt) / Shares Outstanding
"""

def get_ntm_revenue_robust(ticker_obj):
    """Robustly determines NTM Revenue with CAGR fallback."""
    info = ticker_obj.info
    ntm_revenue = info.get('revenueEstimateNextYear')
    
    if ntm_revenue:
        current_rev = info.get('totalRevenue', 1)
        growth_rate = (ntm_revenue / current_rev) - 1
        return ntm_revenue, growth_rate, "Analyst Consensus"

    try:
        # Fallback to 3-Year CAGR if analyst data is missing
        rev_history = ticker_obj.financials.loc['Total Revenue']
        if len(rev_history) >= 3:
            cagr = (rev_history.iloc[0] / rev_history.iloc[2]) ** 0.5 - 1
            source = "3-Year CAGR Fallback"
        else:
            cagr = info.get('revenueGrowth', 0.1)
            source = "Historical Snapshot"
        return info.get('totalRevenue', 0) * (1 + cagr), cagr, source
    except:
        return info.get('totalRevenue', 0) * 1.1, 0.1, "Default 10% Growth"

def calculate_erg_valuation(ticker_symbol, manual_erg=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. Market & Balance Sheet Data
        current_price = info.get('currentPrice')
        shares = info.get('sharesOutstanding')
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        
        # 2. Revenue & Growth Data
        ntm_revenue, growth_rate, rev_source = get_ntm_revenue_robust(ticker)
        growth_pts = growth_rate * 100

        # 3. CALCULATED "LIVE" ERG
        current_ev = (current_price * shares) + net_debt
        current_rev_multiple = current_ev / ntm_revenue
        calculated_erg = current_rev_multiple / growth_pts if growth_pts > 0 else 0

        # 4. TARGET ERG (Override logic)
        target_erg = manual_erg if manual_erg is not None else calculated_erg
        erg_source = "Manual Override" if manual_erg is not None else "Current Market Derived"

        # 5. Fair Value Calculation
        fair_multiple = target_erg * growth_pts
        fair_ev = fair_multiple * ntm_revenue
        fair_price = (fair_ev - net_debt) / shares

        print(f"\n--- ERG REVENUE VALUATION: {ticker_symbol.upper()} ---")
        print(f"Current Price:       ${current_price:.2f}")
        print(f"Growth Basis:        {growth_pts:.1f}% ({rev_source})")
        print("-" * 50)
        print(f"CURRENT ERG RATIO:   {calculated_erg:.2f}")
        print(f"TARGET ERG USED:     {target_erg:.2f} ({erg_source})")
        print("-" * 50)
        print(f"IMPLIED FAIR PRICE:  ${fair_price:.2f}")
        
        upside = (fair_price / current_price) - 1
        print(f"POTENTIAL UPSIDE:    {upside:.1%}")
        print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--erg", type=float, default=None, help="Manual Target ERG Override")
    args = parser.parse_args()
    calculate_erg_valuation(args.ticker, manual_erg=args.erg)