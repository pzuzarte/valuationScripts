import sys
import os
import argparse
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Library equivalent for gross-margin-tiered multiple: valuation_models.run_ev_ntm_revenue(d)
from valuation_models import MARGIN_OF_SAFETY

"""
EV_NTM_rev.py: Growth Stock Valuation Tool (Revenue Multiple Method)

This script calculates the 'Fair Value' of a stock based on its Next Twelve Months 
(NTM) Revenue estimates. It is primarily used for growth companies where 
Enterprise Value (EV) to Revenue is a more appropriate metric than Price-to-Earnings.

WHAT IT DOES:
    1.  Fetches Forward Revenue: Prioritizes analyst consensus estimates for next year's 
        revenue. If unavailable, it calculates a 3-year historical CAGR as a fallback.
    2.  Analyzes Historical Multiples: Calculates a 5-year average EV/Revenue multiple. 
        Unlike basic models, this version dynamically retrieves historical share counts 
        and net debt for each year to ensure the multiple reflects the capital 
        structure at that time.
    3.  Applies Target Multiple: Allows the user to value the stock based on its 
        historical average or a manually defined target multiple.
    4.  Determines Fair Price: Calculates a target Enterprise Value (Multiple * NTM Revenue), 
        subtracts current Net Debt, and divides by current shares to find the intrinsic 
        share price.

HOW TO USE IT:
    Run the script from the terminal with a ticker symbol. You can choose to use 
    the 'historical' average multiple or input your own.

    Example 1 (Use 5-year historical average):
        python EV_NTM_rev.py NVDA --multiple historical

    Example 2 (Use a manual 12x Revenue multiple):
        python EV_NTM_rev.py NVDA --multiple 12

FORMULAS USED:
    - Enterprise Value (EV): Market Cap + Total Debt - Total Cash
    - NTM EV/Revenue: EV / Projected Next 12 Months Revenue
    - Fair Share Price: ((Target Multiple * NTM Revenue) - Net Debt) / Shares Outstanding
"""

def get_historical_multiple(ticker_obj):
    """
    Calculates the 5-year average EV/Revenue multiple with dynamic 
    historical share counts and debt levels.
    """
    try:
        # 1. Fetch Yearly Data
        financials = ticker_obj.get_financials(freq='yearly')
        balance_sheet = ticker_obj.get_balance_sheet(freq='yearly')
        hist = ticker_obj.history(period="7y")['Close']
        
        if financials.empty or 'Total Revenue' not in financials.index:
            return None

        revenues = financials.loc['Total Revenue']
        multiples = []

        for date in revenues.index:
            try:
                # Get the trading price closest to the financial statement date
                target_date = date.tz_localize(None) if date.tzinfo else date
                idx = hist.index.get_indexer([target_date], method='nearest')[0]
                price = hist.iloc[idx]
                
                # Fetch historical capital structure for this specific date
                hist_shares = balance_sheet.loc['Ordinary Shares Number', date] if 'Ordinary Shares Number' in balance_sheet.index else ticker_obj.info.get('sharesOutstanding')
                hist_debt = balance_sheet.loc['Total Debt', date] if 'Total Debt' in balance_sheet.index else 0
                hist_cash = balance_sheet.loc['Cash And Cash Equivalents', date] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                
                rev_val = revenues.loc[date]
                if pd.isna(rev_val) or rev_val <= 0: continue
                
                # Calculate EV using period-specific data
                mkt_cap = price * hist_shares
                ev = mkt_cap + hist_debt - hist_cash
                multiples.append(ev / rev_val)
            except:
                continue

        return sum(multiples) / len(multiples) if multiples else None
    except Exception:
        return None

def calculate_ev_ntm_valuation(ticker_symbol, multiple_input):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. NTM Revenue Data with CAGR Fallback
        ntm_revenue = info.get('revenueEstimateNextYear')
        if not ntm_revenue:
            rev_history = ticker.financials.loc['Total Revenue']
            # Calculate 3-year CAGR
            cagr = (rev_history.iloc[0] / rev_history.iloc[2])**(1/2) - 1 if len(rev_history) >= 3 else 0.10
            ntm_revenue = info.get('totalRevenue', 0) * (1 + cagr)
            source = f"Projected ({cagr:.1%} CAGR fallback)"
        else:
            source = "Analyst Consensus"

        # 2. Historical Analysis
        historical_avg = get_historical_multiple(ticker)
        
        # 3. Determine Target Multiple
        if multiple_input.lower() == 'historical':
            if historical_avg:
                target_multiple = historical_avg
                multiple_note = "5-Year Historical Average"
            else:
                curr_ev = (info.get('marketCap', 0) + info.get('totalDebt', 0) - info.get('totalCash', 0))
                target_multiple = curr_ev / ntm_revenue
                multiple_note = "Current (Historical unavailable)"
        else:
            target_multiple = float(multiple_input)
            multiple_note = f"Manual Target ({target_multiple}x)"

        # 4. Valuation Calculation
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        fair_ev = ntm_revenue * target_multiple
        shares = info.get('sharesOutstanding')
        fair_price = (fair_ev - net_debt) / shares
        current_price = info.get('currentPrice')

        # Formatting Outputs
        print(f"\n--- GROWTH VALUATION: {ticker_symbol.upper()} ---")
        print(f"Current Price:       ${current_price:.2f}")
        print(f"NTM Revenue Est:     ${ntm_revenue/1e9:.2f}B ({source})")
        print("-" * 55)
        print(f"VALUATION MULTIPLE:  {target_multiple:.2f}x")
        print(f"Basis:               {multiple_note}")
        
        if historical_avg:
            diff = ((target_multiple / historical_avg) - 1) * 100
            print(f"vs Historical Avg:   {historical_avg:.2f}x ({diff:+.1f}%)")
            
        print("-" * 55)
        print(f"FAIR VALUE:          ${fair_price:.2f}")
        print(f"POTENTIAL UPSIDE:    {((fair_price/current_price)-1):.1%}")
        print("-" * 55)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--multiple", default="historical")
    args = parser.parse_args()
    calculate_ev_ntm_valuation(args.ticker, args.multiple)