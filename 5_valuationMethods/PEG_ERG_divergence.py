import sys
import os
import argparse
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Library equivalents: valuation_models.run_forward_peg(d), valuation_models.run_erg_valuation(d)
from valuation_models import RISK_FREE_RATE, EQUITY_RISK_PREMIUM

"""
PEG_ERG_divergence.py: Strategic Valuation Divergence Analysis Tool

This script performs a dual-valuation analysis by combining the Forward PEG 
(Earnings-based) and ERG (Revenue-based) methodologies. It is designed to 
identify where a company sits in its business lifecycle and whether its 
market valuation is driven by top-line growth or bottom-line efficiency.

WHAT IT DOES:
    1.  Calculates ERG Fair Value: Values the company based on its revenue 
        growth and a target Enterprise-value-to-Revenue-to-Growth multiple. 
        This highlights the "Land Grab" potential of high-growth tech firms.
    2.  Calculates PEG Fair Value: Values the company based on its forward 
        earnings and a target PEG ratio. This reflects the "Profitability" 
        and efficiency of established businesses.
    3.  Analyzes Divergence: Compares the two fair values. A high ERG relative 
        to PEG suggests an early-stage hyper-growth story, while a high PEG 
        relative to ERG indicates a mature "Profit Optimizer" company.
    4.  Diagnostic Summary: Categorizes the company's status (Convergence, 
        Hyper-Growth, or Profit Optimizer) based on the variance between the 
        two models.

HOW TO USE IT:
    Run the script from the terminal with a ticker and optional target 
    benchmarks for PEG and ERG.

    Example (Standard run for a growth stock):
        python PEG_ERG_divergence.py NVDA

    Example (Custom targets for a mature value stock):
        python PEG_ERG_divergence.py JNJ --peg 1.2 --erg 0.3

FORMULAS:
    - ERG Fair Price = ((Target ERG * Growth * NTM Revenue) - Net Debt) / Shares
    - PEG Fair Price = Target PEG * Forward EPS * (Earnings Growth * 100)
"""

def analyze_valuation(ticker_symbol, target_peg=1.0, target_erg=0.5):
    """
    Combines Forward PEG and ERG valuation methods to identify
    valuation divergence and business lifecycle stages.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # --- 1. DATA GATHERING ---
        current_price = info.get('currentPrice')
        shares = info.get('sharesOutstanding')
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        
        if not current_price or not shares:
            print(f"Error: Basic market data missing for {ticker_symbol}.")
            return

        # Revenue Metrics
        ntm_revenue = info.get('revenueEstimateNextYear') or (info.get('totalRevenue', 0) * 1.1)
        rev_growth = info.get('revenueGrowth', 0.1) # Default 10%
        rev_growth_pts = rev_growth * 100
        
        # Earnings Metrics
        fwd_eps = info.get('forwardEps') or info.get('trailingEps', 0)
        # Prioritize EPS growth estimates, fallback to Revenue growth
        eps_growth_pts = (info.get('earningsQuarterlyGrowth', 0) * 100) or rev_growth_pts
        
        # --- 2. VALUATION CALCULATIONS ---
        # ERG Method (Revenue Based)
        fair_multiple_erg = target_erg * rev_growth_pts
        fv_erg = ((fair_multiple_erg * ntm_revenue) - net_debt) / shares
        
        # PEG Method (Earnings Based) - Logic Guard for Negative Earnings
        if fwd_eps > 0:
            fv_peg = target_peg * fwd_eps * eps_growth_pts
        else:
            fv_peg = None # Cannot value unprofitable companies via PEG
        
        # --- 3. DIVERGENCE ANALYSIS ---
        if fv_peg is not None:
            avg_fv = (fv_erg + fv_peg) / 2
            diff_pct = abs(fv_peg - fv_erg) / avg_fv
        else:
            avg_fv = fv_erg
            diff_pct = 1.0 # Forced divergence due to lack of earnings

        print(f"\n--- VALUATION DASHBOARD: {ticker_symbol.upper()} ---")
        print(f"Current Price: ${current_price:.2f}")
        print("-" * 50)
        peg_display = f"${fv_peg:.2f}" if fv_peg else "N/A (Unprofitable)"
        print(f"PEG Fair Value (Earnings):  {peg_display}")
        print(f"ERG Fair Value (Revenue):   ${fv_erg:.2f}")
        print(f"Average Fair Value:         ${avg_fv:.2f}")
        print(f"Current Upside/Downside:    {((avg_fv/current_price)-1):.1%}")
        print("-" * 50)
        print("DIAGNOSTIC SUMMARY:")

        # --- 4. THE EXPLAINER ENGINE ---
        if fv_peg is None:
            print(">> STATUS: PURE GROWTH STORY")
            print("Company is currently unprofitable. PEG valuation is ignored.")
            print("EXPLANATION: Focus solely on revenue scaling and market share.")

        elif diff_pct < 0.15:
            print(">> STATUS: CONVERGENCE (High Conviction)")
            print("Both Revenue and Earnings growth are perfectly aligned. This is a")
            print("highly efficient business where sales scale directly into profits.")
        
        elif fv_peg > fv_erg * 1.2:
            print(">> STATUS: PROFIT OPTIMIZER (Efficiency Story)")
            print("The 'Bottom Line' is growing faster than the 'Top Line'.")
            print("EXPLANATION: This company is likely raising prices or cutting costs.")
            
        elif fv_erg > fv_peg * 1.2:
            print(">> STATUS: HYPER-GROWTH (Land Grab Story)")
            print("The 'Top Line' is growing much faster than the 'Bottom Line'.")
            print("EXPLANATION: The company is sacrificing short-term profits to scale.")
            
        else:
            print(">> STATUS: MODERATE DIVERGENCE")
            print("Normal variances in accounting. Monitor for trend changes.")

        print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--peg", type=float, default=1.0)
    parser.add_argument("--erg", type=float, default=0.5)
    args = parser.parse_args()
    analyze_valuation(args.ticker, args.peg, args.erg)