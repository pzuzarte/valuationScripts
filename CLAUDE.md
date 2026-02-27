# valuationScripts — Project Guide for Claude

## Project Overview

A comprehensive **stock valuation and portfolio analysis toolkit** for sophisticated retail and professional investors. Provides multiple valuation methodologies, stock screeners, and portfolio-level dashboards. All output is generated as styled HTML reports that auto-open in the browser.

**Primary Language:** Python 3 (no shared modules — each script is fully self-contained)

---

## Project Structure

```
valuationScripts copy/
├── 1_valueScreener/
│   ├── valueScreener.py        # Screens indices for undervalued stocks
│   └── valueData/              # HTML + CSV output
├── 2_growthScreener/
│   ├── growthScreener.py       # Screens indices for high-growth stocks
│   └── growthData/             # HTML + CSV output
├── 3_valuationTool/
│   ├── valuationMaster.py      # Deep single-stock valuation (4-tab report)
│   └── valuationData/          # HTML output
├── 4_portfolioAnalyzer/
│   ├── portfolioAnalyzer.py    # Multi-stock portfolio dashboard
│   ├── FilPortfolio.csv        # Sample portfolio (ticker, shares)
│   └── portfolioData/          # HTML output
└── 5_valuationMethods/         # 10 standalone valuation calculators
    ├── DCFcalc.py
    ├── WACCcalc.py
    ├── forwardDCF.py
    ├── fullDCF.py
    ├── forwardPEG.py
    ├── forwardERG.py
    ├── reverseDCFcalc2.py
    ├── EV_NTM_rev.py
    ├── predictiveERG.py
    └── PEG_ERG_divergence.py
```

---

## Key Scripts

### 1. valueScreener.py (~1,405 lines)
Scans S&P 500 (SPX), Nasdaq 100 (NDX), Russell 2000 (RUT), or TSX for undervalued stocks.

- **Valuation methods:** DCF, P/E, P/FCF, EV/EBITDA, PEG — benchmarked vs live TradingView sector medians
- **Quality filters:** ROE ≥15%, debt limits, value trap detection
- **Scoring:** 0–100 composite score
- **Usage:** `python valueScreener.py --index SPX --top 100 --csv`

### 2. growthScreener.py (~2,597 lines)
Identifies high-growth stocks with upside potential using a four-pillar scoring model.

- **Pillar 1 — Quality (0–35 pts):** ROIC, ROE, margins, FCF, liquidity
- **Pillar 2 — Growth Momentum (0–35 pts):** Revenue/EPS/FCF/GP/NI/EBITDA growth + R&D ratio
- **Pillar 3 — Technical Signal (0–20 pts):** TradingView rating, MoneyFlow, RSI, Stochastic
- **Pillar 4 — Valuation (0–10 pts):** PEG, P/FCF, EV/EBITDA, model upside
- **Price targets:** Blended 60%/40%/20% from Forward PEG, Revenue Extrapolation, EV/EBITDA
- **Usage:** `python growthScreener.py --index NDX --csv`

### 3. valuationMaster.py (~4,973 lines)
Comprehensive single-stock valuation generating a four-tab HTML report.

- **Tab 1 — Classic:** DCF, P/FCF, P/E, EV/EBITDA, convergence analysis
- **Tab 2 — Growth:** Reverse DCF, Forward PEG, EV/NTM Revenue, TAM scenario, Rule of 40
- **Tab 3 — Backtesting (optional):** Historical price tracking accuracy across all methods
- **Tab 4 — Glossary:** Term definitions
- **Usage:** `python valuationMaster.py NVDA --backtest 90 --wacc 0.09`

### 4. portfolioAnalyzer.py (~3,987 lines)
Portfolio-level dashboard for multi-stock analysis.

- **Tabs:** Overview, Signals (BUY/HOLD/SELL), Technical, Fundamentals, Backtesting, Rebalance
- **Rebalance tab:** ATR trade sizing, concentration trims, earnings warnings
- **Input:** CSV with columns `ticker,shares`
- **Usage:** `python portfolioAnalyzer.py FilPortfolio.csv --backtest 90`

### 5. 5_valuationMethods/ (10 standalone modules, ~100–160 lines each)
Quick, interactive calculators for specific valuation methods. Run individually, prompt for input.

| Script | Method |
|--------|--------|
| DCFcalc.py | Classic 2-stage DCF |
| WACCcalc.py | WACC via CAPM |
| forwardDCF.py | DCF + 12-month forward price target |
| fullDCF.py | Bear/Base/Bull DCF + reverse DCF |
| forwardPEG.py | PEG valuation |
| forwardERG.py | ERG (EV/Revenue/Growth) |
| reverseDCFcalc2.py | Revenue ERG model |
| EV_NTM_rev.py | EV/NTM Revenue (5-year avg multiple) |
| predictiveERG.py | Hybrid ERG + Reverse DCF |
| PEG_ERG_divergence.py | PEG vs ERG divergence analysis |

---

## Data Sources & Dependencies

**Required:**
- `yfinance` — Company fundamentals (balance sheets, cash flows, income statements, prices)
- `tradingview_screener` — Live market data, sector medians, technical indicators

**Optional:**
- `pandas`, `numpy` — Enhanced data handling
- `scipy` — Root-finding for reverse DCF

**Standard library used:** `sys`, `os`, `csv`, `json`, `datetime`, `webbrowser`, `argparse`

---

## Financial Assumptions (hardcoded, easily modifiable)

| Constant | Value |
|----------|-------|
| Risk-Free Rate | 4.2–4.3% (10Y Treasury proxy) |
| Equity Risk Premium | 5.5% |
| Terminal Growth Rate | 2–3% |
| DCF Projection Period | 5 years |
| Margin of Safety | 20% |
| Tax Rate | 21% |
| Income Sectors | Utilities, Real Estate, Consumer Staples, Energy |

---

## Code Conventions

- **Architecture:** Functional programming — no classes, no shared modules
- **CLI pattern:** All main scripts use `argparse`
- **Error handling:** Try/except with graceful fallbacks (e.g., analyst estimates → historical CAGR)
- **Output:** Timestamped HTML files (`YYYY_MM_DD_*.html`) auto-opened via `webbrowser`
- **Data retrieval:** Multi-stage fallback logic — analyst consensus first, then historical CAGR
- **DCF formula:** `Fair Value = PV(Stage 1 FCF) + PV(Terminal Value) - Net Debt / Shares Outstanding`

---

## Output Files

| Tool | Output Pattern |
|------|---------------|
| valueScreener | `valueData/YYYY_MM_DD_valuescreen_report.html` + `.csv` |
| growthScreener | `growthData/YYYY_MM_DD_growth_report.html` + `.csv` |
| valuationMaster | `valuationData/YYYY_MM_DD_valuation_TICKER.html` |
| portfolioAnalyzer | `portfolioData/YYYY_MM_DD_portfolio.html` |

---

## Sample Portfolio (FilPortfolio.csv format)

```csv
ticker,shares
NVDA,80
AMZN,50
GOOG,45
META,15
MSFT,15
```

---

## Common Tasks

- **Add a new valuation method:** Create a new standalone script in `5_valuationMethods/` following the same prompt-for-input pattern
- **Modify financial assumptions:** Search for the constant at the top of the relevant script
- **Add a new index to screeners:** Update the index ticker mapping in `valueScreener.py` or `growthScreener.py`
- **Extend HTML reports:** Each script builds HTML via string concatenation — look for `html +=` blocks
