#!/usr/bin/env python3
"""
7_macroDashboard/macroDashboard.py

Macro environment dashboard — tracks the regime context behind every trade:
  · Macro Regime Score  — composite 0-100 risk-on/off indicator (5 components)
  · Rates & Yield Curve — full Treasury curve snapshot + 12-month history
  · Volatility & Credit — VIX, VVIX, HYG vs LQD
  · Sector Rotation     — 11-sector ETF heatmap (1W / 1M / 3M / YTD)
  · Cross-Asset         — SPY, QQQ, IWM, TLT, GLD, Oil, DXY, BTC

Usage:
  python macroDashboard.py

Output: macroData/YYYY_MM_DD_macro.html  (auto-opens in browser)
"""

import argparse
import datetime
import json
import os
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance required.  Run: pip install yfinance")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "macroData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Asset universe ─────────────────────────────────────────────────────────────
SECTORS = {
    "XLK":  "Technology",
    "XLC":  "Comm. Services",
    "XLY":  "Consumer Discret.",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLV":  "Healthcare",
    "XLB":  "Materials",
    "XLE":  "Energy",
    "XLRE": "Real Estate",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
}

CROSS_ASSET = [
    ("SPY",     "S&P 500",        "equity"),
    ("QQQ",     "Nasdaq 100",     "equity"),
    ("IWM",     "Russell 2000",   "equity"),
    ("DIA",     "Dow Jones",      "equity"),
    ("TLT",     "20Y Treasuries", "bond"),
    ("HYG",     "High Yield",     "bond"),
    ("LQD",     "IG Bonds",       "bond"),
    ("GLD",     "Gold",           "commodity"),
    ("USO",     "Crude Oil",      "commodity"),
    ("UUP",     "US Dollar",      "fx"),
    ("BTC-USD", "Bitcoin",        "crypto"),
]

COMMODITIES = [
    ("GLD",  "Gold",         "precious"),
    ("SLV",  "Silver",       "precious"),
    ("USO",  "WTI Oil",      "energy"),
    ("UNG",  "Natural Gas",  "energy"),
    ("CPER", "Copper",       "industrial"),
    ("DBA",  "Agriculture",  "agriculture"),
]

CAP_SIZE = [
    ("SPY",  "S&P 500",         "large"),
    ("MDY",  "S&P MidCap 400",  "mid"),
    ("IWM",  "Russell 2000",    "small"),
    ("IWC",  "Micro-Cap",       "micro"),
]

CAP_STYLE = [
    ("IVW",  "Large Cap Growth", "large-growth"),
    ("IVE",  "Large Cap Value",  "large-value"),
    ("IJK",  "Mid Cap Growth",   "mid-growth"),
    ("IJJ",  "Mid Cap Value",    "mid-value"),
    ("IWO",  "Small Cap Growth", "small-growth"),
    ("IWN",  "Small Cap Value",  "small-value"),
]

GLOBAL_INDICES = [
    ("QQQ",  "Nasdaq 100 (Growth)",       "us"),
    ("SPY",  "S&P 500",                   "us"),
    ("DIA",  "Dow Jones",                 "us"),
    ("IWM",  "Russell 2000 (Small Cap)",  "us"),
    ("VGK",  "Europe",                    "developed"),
    ("EWJ",  "Japan",                     "developed"),
    ("EWG",  "Germany",                   "developed"),
    ("EWU",  "United Kingdom",            "developed"),
    ("EWC",  "Canada",                    "developed"),
    ("EWA",  "Australia",                 "developed"),
    ("EFA",  "Developed ex-US",           "developed"),
    ("MCHI", "China",                     "emerging"),
    ("INDA", "India",                     "emerging"),
    ("EWZ",  "Brazil",                    "emerging"),
    ("EEM",  "Emerging Markets",          "emerging"),
    ("VT",   "All-World",                 "global"),
]

# ── Help popover content ───────────────────────────────────────────────────────
_HELP_DATA = {
    # Overview
    "regime-score": {
        "title": "Macro Regime Score",
        "body": (
            "<b>What it is:</b> A composite 0–100 score summarising the current market environment. "
            "Higher = more risk-on / bullish. Lower = risk-off / defensive.<br><br>"
            "<b>Components (equal weight):</b><br>"
            "· <b>Yield Curve</b> — 10Y–2Y spread; positive spread is risk-on<br>"
            "· <b>VIX</b> — fear gauge; below 20 is calm, above 30 is distressed<br>"
            "· <b>Credit Spreads</b> — HYG vs LQD ratio; tight spreads = healthy credit<br>"
            "· <b>Market Momentum</b> — SPY 1M and 3M performance<br>"
            "· <b>Risk Appetite</b> — SPY 1M vs composite safe-haven (TLT bonds + GLD gold); "
            "equities leading safe havens = risk-on · safe havens leading = risk-off<br><br>"
            "<b>Levels:</b> 0–30 Risk-Off · 30–50 Cautious · 50–70 Neutral · 70–85 Bullish · 85–100 Euphoric"
        ),
    },
    "cross-asset-perf": {
        "title": "Cross-Asset Performance",
        "body": (
            "<b>What it shows:</b> Current price and 1W / 1M / 3M / YTD returns for key macro assets.<br><br>"
            "<b>Assets tracked:</b> SPY (US equities), QQQ (tech/growth), IWM (small-caps), "
            "TLT (long bonds), GLD (gold), USO (crude oil), UNG (natural gas), DXY (US dollar), BTC (crypto).<br><br>"
            "<b>How to read it:</b> When equities and bonds rise together = risk-on with rate cut expectations. "
            "When stocks fall and gold/bonds rise = flight to safety. Dollar strength often weighs on commodities."
        ),
    },
    "cross-asset-chart": {
        "title": "Cross-Asset — 1 Month Return",
        "body": (
            "<b>What it shows:</b> 1-month total return % for each cross-asset, ranked best to worst.<br><br>"
            "<b>Why it matters:</b> Reveals the current rotation theme — e.g. commodity inflation, "
            "flight to safety, or broad risk-on. A chart where equities and gold both lead is rare "
            "and often signals stagflation concerns.<br><br>"
            "<b>Source:</b> Yahoo Finance via yfinance (adjusted close prices)."
        ),
    },
    # Rates & Yield Curve
    "yield-curve-spot": {
        "title": "Yield Curve — Spot (Today)",
        "body": (
            "<b>What it shows:</b> The full US Treasury yield curve from 1-month to 30-year maturities, "
            "sourced from FRED (Federal Reserve Economic Data).<br><br>"
            "<b>Normal curve:</b> Upward sloping — longer maturities yield more. Compensates investors "
            "for duration risk and reflects growth/inflation expectations.<br><br>"
            "<b>Inverted curve:</b> Short-term yields exceed long-term yields. "
            "Has preceded every US recession since 1970, typically by 6–18 months.<br><br>"
            "<b>Flat curve:</b> Transition state — often signals late-cycle slowdown."
        ),
    },
    "spread-history": {
        "title": "10Y – 2Y Spread History (12 Months)",
        "body": (
            "<b>What it is:</b> The 10-year Treasury yield minus the 2-year yield, "
            "the most-watched recession indicator in fixed income.<br><br>"
            "<b>Positive (>0):</b> Normal — growth expected. Historically bullish for risk assets.<br>"
            "<b>Negative (inverted):</b> Fed has tightened too aggressively. "
            "12–18 month recession warning historically. Has inverted before every US recession since 1978.<br><br>"
            "<b>Note:</b> Disinversion (returning from negative to positive) often signals recession "
            "is imminent or already starting — not an all-clear signal.<br><br>"
            "<b>Source:</b> FRED DGS10 – DGS2."
        ),
    },
    "treasury-history": {
        "title": "Treasury Yields — 12-Month History",
        "body": (
            "<b>What it shows:</b> Rolling 12-month history for 3-month, 10-year, and 30-year "
            "US Treasury yields.<br><br>"
            "<b>3-Month (T-Bill):</b> Closely tracks the Fed Funds Rate — moves with Fed policy.<br>"
            "<b>10-Year:</b> The benchmark rate that prices mortgages, corporate bonds, and equity valuations. "
            "Higher 10Y = lower equity P/E multiples justified.<br>"
            "<b>30-Year:</b> Long-duration signal; captures long-run inflation expectations.<br><br>"
            "<b>Source:</b> yfinance (^IRX, ^TNX, ^TYX)."
        ),
    },
    # Volatility & Credit
    "vix-history": {
        "title": "VIX — 12-Month History",
        "body": (
            "<b>What it is:</b> The CBOE Volatility Index — the market's 30-day implied volatility "
            "expectation for the S&amp;P 500, derived from options prices. Often called the 'fear gauge'.<br><br>"
            "<b>Levels:</b><br>"
            "· &lt;15 = Complacency / low fear — often a contrarian warning<br>"
            "· 15–20 = Normal range<br>"
            "· 20–30 = Elevated anxiety<br>"
            "· &gt;30 = Fear / stress<br>"
            "· &gt;40 = Panic / crisis-level<br><br>"
            "<b>Mean-reversion:</b> VIX tends to spike and then revert. Spikes above 30 "
            "historically offer buying opportunities 3–6 months out.<br><br>"
            "<b>Source:</b> yfinance ^VIX."
        ),
    },
    "hyg-lqd": {
        "title": "HYG vs LQD — Credit Spread Proxy",
        "body": (
            "<b>HYG:</b> iShares iBoxx High Yield Corporate Bond ETF — tracks junk bonds (BB and below).<br>"
            "<b>LQD:</b> iShares iBoxx Investment Grade Corporate Bond ETF — tracks investment grade (BBB+).<br><br>"
            "<b>Why the ratio matters:</b> When HYG underperforms LQD, credit conditions are tightening — "
            "investors are demanding more premium for credit risk. "
            "This typically leads equity stress by 1–4 weeks.<br><br>"
            "<b>Normalized to 100:</b> Both series rebased to 100 at the start of the window "
            "to highlight relative performance.<br><br>"
            "<b>Note:</b> Interest rate changes affect both ETFs similarly; "
            "divergence between them isolates the credit risk premium."
        ),
    },
    "vix-guide": {
        "title": "VIX Interpretation Guide",
        "body": (
            "<b>Practical uses of VIX:</b><br><br>"
            "· <b>Options pricing:</b> High VIX = expensive options. Better to sell premium (covered calls, "
            "cash-secured puts). Low VIX = cheap options. Better to buy protection or speculative calls.<br><br>"
            "· <b>Position sizing:</b> Scale down risk when VIX is elevated; market moves larger in both directions.<br><br>"
            "· <b>Contrarian signal:</b> VIX spikes above 35–40 have historically been near-term "
            "buying opportunities in the S&amp;P 500 (3–6 month horizon).<br><br>"
            "· <b>VVIX (vol of vol):</b> Measures how fast VIX itself is moving. "
            "High VVIX means uncertainty about uncertainty — extreme positioning, trade with caution."
        ),
    },
    # Sector Rotation
    "sector-perf": {
        "title": "Sector Performance — All Periods",
        "body": (
            "<b>What it shows:</b> Return heatmap for the 11 GICS sectors via SPDR ETFs "
            "(XLK, XLC, XLY, XLF, XLI, XLV, XLB, XLE, XLRE, XLP, XLU) across "
            "1W / 1M / 3M / YTD periods.<br><br>"
            "<b>How to read rotation:</b><br>"
            "· <b>Early cycle:</b> Financials, Consumer Discretionary, Industrials lead<br>"
            "· <b>Mid cycle:</b> Technology, Comm. Services, Materials lead<br>"
            "· <b>Late cycle:</b> Energy, Industrials, Healthcare lead<br>"
            "· <b>Recession:</b> Utilities, Consumer Staples, Healthcare lead (defensives)<br><br>"
            "<b>Cross-period consistency:</b> A sector leading on 1W, 1M, and 3M all at once "
            "has strong momentum. A sector leading only on 1W but lagging 3M = short-term bounce."
        ),
    },
    "rotation-quadrant": {
        "title": "Rotation Quadrant — 1M vs 3M Return",
        "body": (
            "<b>What it shows:</b> Each sector plotted with 3M return (X-axis) vs 1M return (Y-axis).<br><br>"
            "<b>Quadrants:</b><br>"
            "· <b>Top-right (Improving):</b> Strong 3M and accelerating 1M — momentum leaders<br>"
            "· <b>Top-left (Lagging → Recovering):</b> Weak 3M but positive recent 1M — early rotation candidates<br>"
            "· <b>Bottom-right (Leading → Weakening):</b> Strong 3M but fading 1M — potential rotation out<br>"
            "· <b>Bottom-left (Lagging):</b> Weak on both — avoid or short<br><br>"
            "<b>Investment application:</b> Buy sectors moving from top-left to top-right. "
            "Watch for distribution in bottom-right."
        ),
    },
    "momentum-bar": {
        "title": "1-Month Momentum — Ranked Best to Worst",
        "body": (
            "<b>What it shows:</b> Sector ETFs ranked by 1-month return, best at top.<br><br>"
            "<b>How to use it:</b> Momentum investing favours the top 3 sectors; "
            "mean-reversion strategies watch for extreme bottom performers.<br><br>"
            "<b>Caution:</b> 1-month is short — single events (earnings, macro shocks) can "
            "temporarily distort rankings. Cross-reference with the 3M column in the table.<br><br>"
            "<b>Colour:</b> Green = positive return. Red = negative return."
        ),
    },
    # Macro Signals
    "buffett-index": {
        "title": "Buffett Index (Market Cap / GDP)",
        "body": (
            "<b>What it is:</b> Total US stock market capitalisation as a percentage of US GDP — "
            "Warren Buffett's preferred broad market valuation measure.<br><br>"
            "<b>Calculation:</b> Wilshire 5000 index (^W5000, proxy for total market cap in $B) "
            "÷ US nominal GDP (FRED, quarterly) × 100.<br><br>"
            "<b>Levels (approximate):</b><br>"
            "· &lt;75% = Undervalued (buy zone)<br>"
            "· 75–90% = Fair Value<br>"
            "· 90–115% = Modestly Overvalued<br>"
            "· 115–140% = Significantly Overvalued<br>"
            "· &gt;140% = Extreme — territory seen only in dot-com peak and post-2020<br><br>"
            "<b>Limitations:</b> Doesn't account for foreign earnings of US multinationals, "
            "low interest rate environments, or structural shifts in the economy. "
            "Best used as a long-run valuation anchor, not a timing tool."
        ),
    },
    "shiller-cape": {
        "title": "Shiller CAPE Ratio (P/E10)",
        "body": (
            "<b>What it is:</b> Cyclically Adjusted Price-to-Earnings ratio — "
            "S&amp;P 500 price divided by the average of the last 10 years of real (inflation-adjusted) earnings. "
            "Developed by Nobel laureate Robert Shiller.<br><br>"
            "<b>Why 10-year average:</b> Smooths out business cycle swings in earnings, "
            "giving a more stable valuation signal than trailing 12-month P/E.<br><br>"
            "<b>Historical average:</b> ~17× (1880–present). Since 1990: ~27×.<br><br>"
            "<b>Levels:</b><br>"
            "· &lt;15× = Undervalued<br>"
            "· 15–22× = Fair Value<br>"
            "· 22–30× = Elevated<br>"
            "· 30–40× = Expensive (top decile historically)<br>"
            "· &gt;40× = Extreme — dot-com peak was 44×<br><br>"
            "<b>Predictive power:</b> High CAPE predicts lower 10-year forward returns. "
            "CAPE above 30 has historically implied &lt;3% real annual returns over the next decade.<br><br>"
            "<b>Source:</b> multpl.com (Robert Shiller data)."
        ),
    },
    "recession-prob": {
        "title": "NY Fed Recession Probability (12M)",
        "body": (
            "<b>What it is:</b> The New York Federal Reserve's model probability of a US recession "
            "within the next 12 months, based on the Treasury yield curve slope (10Y–3M spread).<br><br>"
            "<b>How it works:</b> A probit regression model fitted on post-WWII data. "
            "The yield curve inversion is the model's primary input — deeper inversion → higher probability.<br><br>"
            "<b>Signal levels:</b><br>"
            "· &lt;20% = Low risk<br>"
            "· 20–40% = Elevated — worth watching<br>"
            "· &gt;40% = High — historically associated with recessions within 12 months<br>"
            "· &gt;60% = Very high — recession nearly certain (was 83% before 2008, 43% before 2001)<br><br>"
            "<b>Lag:</b> Published monthly with ~3-month data lag.<br><br>"
            "<b>Source:</b> FRED RECPROUSM156N."
        ),
    },
    "hy-oas": {
        "title": "Credit Spreads — HY OAS & IG OAS",
        "body": (
            "<b>OAS (Option-Adjusted Spread):</b> The yield premium that corporate bonds pay "
            "over equivalent-maturity Treasuries, adjusted for embedded options (e.g. call features).<br><br>"
            "<b>HY OAS (High Yield):</b> Junk bond spread (ICE BofA US High Yield Index, BAMLH0A0HYM2). "
            "Normal: 3–4%. Stress: 5–7%. Crisis: &gt;8%.<br><br>"
            "<b>IG OAS (Investment Grade):</b> Investment-grade corporate spread (BAMLC0A0CM). "
            "Normal: 0.7–1.2%. Stress: 1.5–2.5%.<br><br>"
            "<b>Why it matters:</b> Credit markets often lead equities. "
            "Widening spreads signal corporate stress before it appears in earnings. "
            "Tightening spreads confirm risk appetite is healthy.<br><br>"
            "<b>Source:</b> FRED (ICE BofA / Bank of America data)."
        ),
    },
    "credit-spread-chart": {
        "title": "HY & IG Credit Spreads — 20-Year History",
        "body": (
            "<b>What it shows:</b> Two decades of daily HY OAS and IG OAS, "
            "allowing regime comparison to 2008 GFC, 2020 COVID, and 2022 rate-shock periods.<br><br>"
            "<b>Key episodes to observe:</b><br>"
            "· <b>2008–09:</b> HY spread reached 22%, IG hit 6% — systemic credit crisis<br>"
            "· <b>2020 Mar:</b> HY spiked to 11% in weeks, then recovered as Fed intervened<br>"
            "· <b>2022:</b> Modest widening on rate shock but no credit crisis<br><br>"
            "<b>Current context:</b> Tight spreads (HY &lt;4%) indicate credit markets are relaxed. "
            "If HY starts rising while equities hold, credit is likely leading equities lower."
        ),
    },
    "buffett-chart": {
        "title": "Buffett Index — 10-Year History",
        "body": (
            "<b>What it shows:</b> Quarterly Buffett Index (Market Cap / GDP %) over 10 years.<br><br>"
            "<b>Data sources:</b> Wilshire 5000 (^W5000 via Yahoo Finance, quarterly sampled) "
            "÷ US nominal GDP (FRED GDP, forward-filled to match quarters) × 100.<br><br>"
            "<b>Key thresholds shown:</b> 100% = GDP parity. Historical peaks: ~160% (2021), "
            "~148% (dot-com 2000). Current values above 150% are historically rare.<br><br>"
            "<b>Note:</b> The index has drifted higher structurally since 2010 due to "
            "globalisation of US corporate earnings and lower discount rates."
        ),
    },
    "cape-chart": {
        "title": "Shiller CAPE — History (Since 1990)",
        "body": (
            "<b>What it shows:</b> Monthly CAPE ratio from 1990 to today. "
            "The dashed line marks the long-run average for the shown period.<br><br>"
            "<b>Key historical reference points:</b><br>"
            "· 1999–2000: CAPE peaked at ~44× (dot-com bubble)<br>"
            "· 2009 trough: ~13× (post-GFC)<br>"
            "· 2022 rate-shock correction: fell from ~38× to ~28×<br><br>"
            "<b>What to watch:</b> Sustained readings above 30× compress forward returns. "
            "Mean reversion can take years — elevated CAPE is a 7–10 year signal, not a market timing tool.<br><br>"
            "<b>Source:</b> multpl.com (Robert Shiller / Yale data)."
        ),
    },
    "recession-chart": {
        "title": "NY Fed 12-Month Recession Probability",
        "body": (
            "<b>What it shows:</b> Monthly time series of the NY Fed's yield-curve-based "
            "recession probability model going back to 2000.<br><br>"
            "<b>False positives:</b> The model flagged elevated probability in 2019 "
            "(inversion without immediate recession due to Fed pivot). "
            "It is a probabilistic tool, not a certainty.<br><br>"
            "<b>Lead time:</b> Historically provides 6–18 months of lead time before a recession "
            "— enough to reduce equity exposure and extend bond duration.<br><br>"
            "<b>Disinversion watch:</b> When the spread re-steepens after inversion, "
            "the probability falls — but that phase historically marks recession onset."
        ),
    },
    "money-markets": {
        "title": "US Money Market Fund AUM",
        "body": (
            "<b>What it is:</b> Total assets held in US money market mutual funds — "
            "a proxy for 'cash on the sidelines'.<br><br>"
            "<b>Why it matters:</b> During risk-off periods, investors shift capital into money market funds "
            "for safety and yield. A large AUM overhang is often cited as potential fuel "
            "for a future equity rally when sentiment turns.<br><br>"
            "<b>Counter-argument:</b> High MMF assets can persist for years — not a short-term timing tool. "
            "The relevant signal is the rate of change: rapid outflows → risk-on rotation; "
            "rapid inflows → risk-off flight.<br><br>"
            "<b>Source:</b> FRED MMMFFAQ027S (Federal Reserve, quarterly)."
        ),
    },
    "us-vs-intl": {
        "title": "US vs International Equity Rotation",
        "body": (
            "<b>What it shows:</b> 1-year price performance of SPY (US), EFA (Developed ex-US), "
            "and EEM (Emerging Markets), plus the SPY/EFA ratio trend.<br><br>"
            "<b>SPY/EFA ratio:</b> Rising = US outperforming developed markets. "
            "Falling = international rotation underway.<br><br>"
            "<b>Why international matters:</b> US equities have outperformed since 2010, "
            "but mean-reversion episodes occur. International stocks typically outperform "
            "when: (1) USD weakens, (2) emerging market growth accelerates, "
            "(3) US valuations are stretched relative to ex-US.<br><br>"
            "<b>EEM vs EFA:</b> EEM (emerging) has higher growth potential but more volatility "
            "and currency risk. EFA (developed) is more stable but slower-growing."
        ),
    },
    # Commodities
    "commodity-perf": {
        "title": "Commodity ETF Performance",
        "body": (
            "<b>ETFs tracked:</b><br>"
            "· <b>GLD</b> — SPDR Gold Shares (gold bullion)<br>"
            "· <b>SLV</b> — iShares Silver Trust (silver bullion)<br>"
            "· <b>USO</b> — United States Oil Fund (WTI crude oil futures)<br>"
            "· <b>UNG</b> — United States Natural Gas Fund (Henry Hub natural gas)<br>"
            "· <b>CPER</b> — United States Copper Index Fund (copper futures)<br>"
            "· <b>DBA</b> — Invesco DB Agriculture Fund (diversified agricultural commodities)<br><br>"
            "<b>Why commodities matter:</b> Leading indicators of inflation, global growth, "
            "and supply shocks. Used by central banks, fund managers, and macro traders "
            "to anticipate policy moves and sector rotations.<br><br>"
            "<b>Note:</b> Futures-based ETFs (USO, UNG, CPER, DBA) suffer from roll yield "
            "in contango markets — long-term holding costs can differ from spot price moves."
        ),
    },
    "commodity-normalized": {
        "title": "All Commodities — 1Y Normalized",
        "body": (
            "<b>What it shows:</b> All 6 commodity ETFs rebased to 100 one year ago, "
            "allowing direct performance comparison on a single chart.<br><br>"
            "<b>How to read it:</b> Lines above 100 have gained. Lines below have lost. "
            "The spread between the highest and lowest lines shows commodity dispersion.<br><br>"
            "<b>Macro signals from relative performance:</b><br>"
            "· Gold leading = risk-off / dollar weakness / inflation fears<br>"
            "· Oil and copper leading = strong global growth demand<br>"
            "· Agriculture leading = supply shock or food inflation<br>"
            "· Natural gas diverging from oil = regional energy dynamics"
        ),
    },
    "gold-copper": {
        "title": "Gold / Copper Ratio — Risk Sentiment",
        "body": (
            "<b>What it is:</b> Gold price divided by copper price (via GLD/CPER ETF ratio).<br><br>"
            "<b>Why it matters:</b> Gold is a safe-haven asset. Copper is the 'Dr. Copper' "
            "industrial metal highly correlated with global economic activity.<br><br>"
            "<b>Rising ratio:</b> Gold outperforming copper → risk-off signal. "
            "Markets pricing in economic slowdown or geopolitical stress.<br><br>"
            "<b>Falling ratio:</b> Copper outperforming gold → risk-on signal. "
            "Markets pricing in accelerating industrial demand and growth.<br><br>"
            "<b>Historical use:</b> Historically leads equity market turns by 1–3 months "
            "at major cycle inflection points."
        ),
    },
    "gold-silver": {
        "title": "Gold / Silver Ratio",
        "body": (
            "<b>What it is:</b> Gold price divided by silver price (GLD/SLV ETF ratio).<br><br>"
            "<b>Normal range:</b> 50–80×. Historically averages ~65–70×.<br><br>"
            "<b>High ratio (&gt;80×):</b> Silver relatively cheap vs gold — "
            "historically associated with recessions and risk-off periods. "
            "Silver tends to outperform in recovery (industrial demand picks up).<br><br>"
            "<b>Low ratio (&lt;50×):</b> Silver expensive relative to gold — "
            "often seen during industrial booms and high-inflation periods.<br><br>"
            "<b>Note:</b> Silver has dual demand — ~50% industrial (electronics, solar panels), "
            "~50% investment. It is more volatile and more cyclical than gold."
        ),
    },
    "oil-gas": {
        "title": "Oil vs Natural Gas — Relative Energy",
        "body": (
            "<b>What it shows:</b> WTI crude oil (USO) and natural gas (UNG) rebased to 100 "
            "one year ago — relative performance of two key energy benchmarks.<br><br>"
            "<b>Why they diverge:</b> Oil is a globally priced commodity tied to transport and "
            "geopolitics. Natural gas is more regionally priced (US Henry Hub), driven by "
            "heating/cooling demand, LNG exports, and domestic storage.<br><br>"
            "<b>Macro read:</b><br>"
            "· Both rising = energy inflation / supply constraint<br>"
            "· Oil up, gas flat = global demand without domestic shortage<br>"
            "· Gas up, oil flat = regional supply squeeze (e.g. cold winter, LNG export surge)<br>"
            "· Both falling = demand destruction / economic slowdown"
        ),
    },
    "commodity-signals": {
        "title": "Commodity Macro Signals Table",
        "body": (
            "<b>What it shows:</b> Each commodity's traditional macro interpretation — "
            "what rising or falling prices historically signal about the economy.<br><br>"
            "<b>How to use it:</b> Look for confirmation across multiple commodities. "
            "If gold and bonds are both rising while copper and oil fall, "
            "that's a strong risk-off, slowdown signal. Mixed signals reduce conviction.<br><br>"
            "<b>Important:</b> Commodity signals are macro context tools, not trading signals. "
            "Supply shocks (wars, weather, OPEC decisions) can temporarily override economic signals."
        ),
    },
    # Cap Size & Style
    "cap-size": {
        "title": "Cap Size Rotation",
        "body": (
            "<b>What it shows:</b> Price performance across the four market-cap tiers: "
            "Large (SPY), Mid (MDY), Small (IWM), and Micro (IWC).<br><br>"
            "<b>Why it matters:</b> Small-cap outperformance signals broad economic confidence — "
            "investors are willing to take on more risk. Small-cap underperformance signals "
            "defensiveness or credit stress (small caps are more leveraged and rate-sensitive).<br><br>"
            "<b>Classic signal:</b> IWM leading SPY by 3%+ over 3 months = healthy bull market breadth. "
            "IWM lagging by 4%+ = narrowing leadership, caution flag."
        ),
    },
    "cap-style": {
        "title": "Growth vs Value — Across Cap Sizes",
        "body": (
            "<b>What it shows:</b> Returns for growth and value ETFs at each cap tier, "
            "covering Large, Mid, and Small.<br><br>"
            "<b>Growth vs Value rotation:</b> Growth outperforms when rates are falling or expected to fall "
            "(future earnings worth more at lower discount rates). Value outperforms when rates rise, "
            "inflation is elevated, or the economic cycle is early-to-mid stage.<br><br>"
            "<b>Key ratio to watch:</b> IVW/IVE (Large Cap Growth/Value). "
            "Rising = growth regime. Falling = value rotation. "
            "This ratio closely tracks the 10Y Treasury yield — rising yields compress growth multiples."
        ),
    },
    "cap-size-chart": {
        "title": "Cap Size — 1Y Normalised Chart",
        "body": (
            "<b>What it shows:</b> SPY, MDY, IWM, and IWC all set to 100 at the start of the "
            "12-month window, so you can directly compare relative performance across cap tiers.<br><br>"
            "<b>How to read it:</b> A widening gap between large (SPY) and small (IWM/IWC) suggests "
            "mega-cap concentration driving index returns. Converging lines = broader participation. "
            "IWC leading everything = very risk-on environment."
        ),
    },
    "growth-value-ratio": {
        "title": "Growth / Value Ratio (IVW ÷ IVE)",
        "body": (
            "<b>What it shows:</b> The price ratio of iShares Large Cap Growth (IVW) divided by "
            "iShares Large Cap Value (IVE) over the past 12 months.<br><br>"
            "<b>Rising ratio:</b> Growth outperforming value — typically driven by falling rate "
            "expectations, strong tech earnings, or risk-on sentiment.<br><br>"
            "<b>Falling ratio:</b> Value rotation — often driven by rising rates, early economic "
            "cycle conditions, inflation, or a shift from momentum to cheapness.<br><br>"
            "<b>Historical context:</b> Growth dramatically outperformed value 2017–2021 (zero rates). "
            "Value rebounded sharply in 2022 as rates surged."
        ),
    },
    # Global Indices
    "global-indices": {
        "title": "Global Market Indices",
        "body": (
            "<b>What it shows:</b> Performance across 16 major global ETFs spanning US, "
            "Developed, Emerging, and All-World markets — sorted by 1-month return.<br><br>"
            "<b>Region colour codes:</b> Blue = US · Green = Developed ex-US · "
            "Orange = Emerging Markets · Purple = All-World.<br><br>"
            "<b>How to use it:</b> Look for regional leadership rotation. "
            "When EFA or EEM outperform SPY, it often signals USD weakness, "
            "relative valuation opportunities abroad, or improving global growth. "
            "US dominance (QQQ/SPY leading) signals dollar strength and US earnings premium."
        ),
    },
    "global-norm-chart": {
        "title": "QQQ vs Global — 1Y Normalised Chart",
        "body": (
            "<b>What it shows:</b> Key global indices all set to 100 at the start of the "
            "12-month window. QQQ (Nasdaq growth) is shown in bold blue as the benchmark.<br><br>"
            "<b>How to read it:</b> Lines above QQQ mean that region is outperforming US growth. "
            "Lines below = underperformance vs the Nasdaq benchmark.<br><br>"
            "<b>Why QQQ as benchmark?</b> QQQ represents the highest-returning US equity benchmark "
            "of the past decade. Any global region that outperforms it deserves serious attention "
            "as a potential rotation target."
        ),
    },
    "global-bar-chart": {
        "title": "Global 1M Returns — Ranked",
        "body": (
            "<b>What it shows:</b> 1-month return for every tracked global index, "
            "sorted best to worst from top to bottom.<br><br>"
            "<b>How to use it:</b> Quickly identify which geographies are leading and lagging "
            "the current month. Persistent leadership from the same region across multiple months "
            "is a rotation signal worth acting on."
        ),
    },
}


def help_btn(key: str) -> str:
    """Inline ⓘ button that opens the help popover for the given key."""
    safe = key.replace("'", "\\'")
    return (
        f'<button class="help-btn" onclick="showHelp(\'{safe}\',event)" '
        f'aria-label="Help" title="Click for more info">ⓘ</button>'
    )


# yfinance yield tickers used for historical charts
YF_YIELDS = {
    "3M":  "^IRX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

# US Treasury XML API field mapping (no API key needed)
# Source: home.treasury.gov daily yield curve
TREASURY_XML_MAP = {
    "1M":  "BC_1MONTH",   # 4-week T-bill (field present from ~2018)
    "3M":  "BC_3MONTH",
    "6M":  "BC_6MONTH",
    "1Y":  "BC_1YEAR",
    "2Y":  "BC_2YEAR",
    "3Y":  "BC_3YEAR",
    "5Y":  "BC_5YEAR",
    "7Y":  "BC_7YEAR",
    "10Y": "BC_10YEAR",
    "20Y": "BC_20YEAR",
    "30Y": "BC_30YEAR",
}

# Keep FRED_YIELDS as an alias so nothing else breaks
FRED_YIELDS = [(label, field) for label, field in TREASURY_XML_MAP.items()]

# Ordered maturity labels (used in charts / tables)
TREASURY_FIELDS = [(label, None) for label in TREASURY_XML_MAP]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def pct_chg(new, old):
    if old and old != 0:
        return round((new - old) / abs(old) * 100, 2)
    return None

def fmt_pct(v, plus=True):
    if v is None:
        return "—"
    sign = "+" if (plus and v >= 0) else ""
    return f"{sign}{v:.2f}%"

def clr(v):
    """Green/red/muted based on sign."""
    if v is None:
        return "#6b7194"
    return "#00c896" if v >= 0 else "#e05c5c"


def _live1d_cell(ticker, v):
    """TD element for the live 1D column.

    Embeds the statically-fetched value (populated at script run-time) and
    carries the ``live1d`` CSS class + ``data-ticker`` attribute so the
    in-browser JavaScript can overwrite it with a fresher value when the
    Flask server is running.
    """
    if v is None:
        return (f'<td class="num live1d" data-ticker="{ticker}" '
                f'style="color:#6b7194">—</td>')
    alpha = min(0.28, abs(v) / 5 * 0.28)
    bg    = f"rgba(0,200,150,{alpha:.3f})" if v >= 0 else f"rgba(224,92,92,{alpha:.3f})"
    return (f'<td class="num live1d" data-ticker="{ticker}" '
            f'style="color:{clr(v)};font-weight:600;background:{bg}">'
            f'{fmt_pct(v)}</td>')


# ── Interpretation helpers ──────────────────────────────────────────────────
def _b(txt):
    return f'<span style="color:var(--up);font-weight:600">{txt}</span>'

def _r(txt):
    return f'<span style="color:var(--down);font-weight:600">{txt}</span>'

def _w(txt):
    return f'<span style="color:var(--warn);font-weight:600">{txt}</span>'

def _v(txt):
    return f'<strong>{txt}</strong>'

def _interp_card(title: str, bullets: list) -> str:
    """Analyst write-up card rendered at the bottom of a dashboard tab."""
    if not bullets:
        return ""
    h = f'<div class="interp-card"><h4>{title}</h4><ul class="interp-list">'
    for b in bullets:
        h += f'<li>{b}</li>'
    h += '</ul></div>'
    return h


def _interp_overview(regime: dict, cross: dict) -> str:
    bullets = []
    score = regime["score"]
    comps = regime.get("components", {})

    # Overall regime
    if score >= 80:    mood = _b("strongly risk-on")
    elif score >= 62:  mood = _b("moderately bullish")
    elif score >= 42:  mood = _w("neutral / mixed")
    elif score >= 25:  mood = _r("cautious / risk-off tilt")
    else:              mood = _r("risk-off")
    bullets.append(f'The Macro Regime Score is {_v(f"{score}/100")} ({mood}). All five components are factored in — a score below 42 warrants defensive positioning.')

    # Component strengths / drags
    strong = [c["label"] for c in comps.values() if c.get("score") is not None and c["score"] >= 14]
    weak   = [c["label"] for c in comps.values() if c.get("score") is not None and c["score"] < 8]
    if strong:
        bullets.append(f'Constructive signals: {_b(", ".join(strong))}.')
    if weak:
        bullets.append(f'Drag on the score: {_r(", ".join(weak))} — these are in cautionary territory and worth monitoring.')

    # SPY 1M
    spy    = cross.get("SPY", {})
    spy_1m = spy.get("1M")
    spy_vs200 = spy.get("vs_200")
    if spy_1m is not None:
        if spy_1m > 2:
            bullets.append(f'The S&P 500 (SPY) returned {_b(fmt_pct(spy_1m))} over the past month, confirming near-term bullish price action.')
        elif spy_1m > -2:
            bullets.append(f'The S&P 500 (SPY) returned {_w(fmt_pct(spy_1m))} over the past month — essentially flat, consistent with a neutral/consolidating market.')
        else:
            bullets.append(f'The S&P 500 (SPY) fell {_r(fmt_pct(spy_1m))} over the past month, reflecting genuine selling pressure.')
    if spy_vs200 is not None:
        if spy_vs200 > 0:
            bullets.append(f'SPY is trading {_b(fmt_pct(spy_vs200))} above its 200-day moving average — the primary trend remains up.')
        else:
            bullets.append(f'SPY is {_r(fmt_pct(spy_vs200))} below its 200-day MA — the primary uptrend has broken, a meaningful caution signal.')

    # Gold vs equities
    gld_1m = cross.get("GLD", {}).get("1M")
    if gld_1m is not None and spy_1m is not None:
        if gld_1m > spy_1m + 3:
            bullets.append(f'Gold ({_b(fmt_pct(gld_1m))} 1M) is significantly outperforming equities — a classic flight-to-safety pattern suggesting elevated risk aversion.')
        elif spy_1m > gld_1m + 3:
            bullets.append(f'Equities ({_b(fmt_pct(spy_1m))} 1M) are outpacing gold ({fmt_pct(gld_1m)}) — capital is rotating into risk assets.')

    # Breadth (IWM vs SPY 3M)
    iwm_3m = cross.get("IWM", {}).get("3M")
    spy_3m = cross.get("SPY", {}).get("3M")
    if iwm_3m is not None and spy_3m is not None:
        diff = iwm_3m - spy_3m
        if diff > 2:
            bullets.append(f'Small-caps (IWM) are outperforming large-caps (SPY) over 3 months — broad participation supports the bull case.')
        elif diff < -4:
            bullets.append(f'Small-caps lag large-caps by {_r(f"{abs(diff):.1f}pp")} over 3 months — narrow leadership is a health warning for this rally.')

    return _interp_card("Analyst Read — Overview", bullets)


def _interp_rates(treasury: dict, yf_yields: dict) -> str:
    bullets = []
    cur = treasury.get("current", {})
    y10 = cur.get("10Y")
    y2  = cur.get("2Y")
    y3m = cur.get("3M")

    if y10 is None:
        return ""

    # 10Y level
    if y10 > 5.0:
        bullets.append(f'The 10-year yield is {_r(f"{y10:.2f}%")} — historically restrictive. High rates compress equity valuations (higher discount rate) and increase financing costs economy-wide.')
    elif y10 > 4.0:
        bullets.append(f'The 10-year yield stands at {_w(f"{y10:.2f}%")} — elevated by post-GFC standards. The equity risk premium has compressed, making bonds a legitimate income alternative for the first time in years.')
    elif y10 > 2.5:
        bullets.append(f'The 10-year yield is {_b(f"{y10:.2f}%")} — within a broadly neutral range. Rates are not an acute headwind to equity multiples at current levels.')
    else:
        bullets.append(f'The 10-year yield is {_b(f"{y10:.2f}%")} — low, which historically supports equity multiple expansion and growth asset valuations.')

    # Yield curve shape
    if y2 is not None:
        spread = round(y10 - y2, 2)
        if spread < -0.5:
            bullets.append(f'The 10Y–2Y curve is {_r(f"inverted ({spread:+.2f}%)")} — a historically reliable (if lagged) recession warning. Every US recession since 1955 has been preceded by yield-curve inversion.')
        elif spread < 0:
            bullets.append(f'The yield curve is mildly {_w(f"inverted ({spread:+.2f}%)")}. Mild inversion warrants monitoring; sustained or deepening inversion materially increases recession odds.')
        elif spread < 0.5:
            bullets.append(f'The yield curve is {_w(f"nearly flat ({spread:+.2f}%)")} — consistent with a mature economic cycle or uncertainty around the Fed\'s next move.')
        else:
            bullets.append(f'The yield curve has a {_b(f"positive slope ({spread:+.2f}%)")} — a normal, healthy configuration supportive of bank lending and economic expansion.')

    # 10Y–3M spread (NY Fed preferred measure)
    if y3m is not None:
        spread_3m = round(y10 - y3m, 2)
        if spread_3m < -0.25:
            bullets.append(f'The 10Y–3M spread ({_r(f"{spread_3m:+.2f}%")}) is negative — the NY Fed\'s preferred recession-signal measure. This inversion is the primary input into the NY Fed recession model.')

    # Equity valuation impact
    if y10 > 4.5:
        bullets.append(f'At {y10:.2f}%, fixed income now provides meaningful competition to equities. A 10Y above 4.5% historically keeps a lid on P/E multiple expansion.')

    return _interp_card("Analyst Read — Rates & Yield Curve", bullets)


def _interp_volatility(vix: dict, cross: dict, credit: dict) -> str:
    bullets = []
    vix_val = vix.get("current")
    vix_avg = vix.get("avg_1y")
    vvix    = vix.get("vvix")

    if vix_val is None:
        return ""

    # VIX level
    if vix_val < 15:
        bullets.append(f'VIX is {_b(f"{vix_val:.1f}")} — low, signalling market complacency. Sustained sub-15 VIX environments can persist in bull markets but often precede larger vol spikes; asymmetric downside risk is elevated.')
    elif vix_val < 20:
        bullets.append(f'VIX is {_b(f"{vix_val:.1f}")} — within the normal range. Investors are not paying a premium for protection; market sentiment is broadly constructive.')
    elif vix_val < 28:
        bullets.append(f'VIX is {_w(f"{vix_val:.1f}")} — elevated. Uncertainty is rising; institutional hedging is increasing. Equity dip-buyers should be selective.')
    elif vix_val < 40:
        bullets.append(f'VIX is {_r(f"{vix_val:.1f}")} — high fear territory. Significant institutional hedging is underway. Historically, readings above 30 have been excellent contrarian entry points 6–12 months forward.')
    else:
        bullets.append(f'VIX is {_r(f"{vix_val:.1f}")} — crisis-level volatility. Past instances (COVID-19, GFC, post-9/11) have historically marked excellent long-term buying opportunities for patient capital.')

    # VIX vs 1Y average
    if vix_avg:
        vs_avg = round(vix_val - vix_avg, 1)
        if vs_avg > 5:
            bullets.append(f'VIX is {_r(f"{vs_avg:+.1f} pts")} above its 1-year average ({vix_avg:.1f}) — meaningfully elevated. Current fear is well above the recent regime baseline.')
        elif vs_avg < -3:
            bullets.append(f'VIX is {_b(f"{vs_avg:+.1f} pts")} below its 1-year average — unusually calm relative to recent history. Watch for vol mean reversion.')
        else:
            bullets.append(f'VIX is near its 1-year average ({vix_avg:.1f}) — current volatility is in line with recent norms, neither stressed nor complacent.')

    # VVIX
    if vvix:
        if vvix > 120:
            bullets.append(f'VVIX (vol-of-vol) is {_r(f"{vvix:.0f}")} — extreme. Options on VIX are very expensive; expect outsized VIX swings in either direction. Tail hedges are crowded.')
        elif vvix > 100:
            bullets.append(f'VVIX is {_w(f"{vvix:.0f}")} — elevated. The options market is pricing in continued VIX uncertainty; tail-risk protection demand is above average.')
        else:
            bullets.append(f'VVIX is {_b(f"{vvix:.0f}")} — benign. Vol-of-vol is calm, consistent with a relatively stable near-term volatility regime.')

    # HYG vs LQD credit divergence
    hyg_1m = cross.get("HYG", {}).get("1M")
    lqd_1m = cross.get("LQD", {}).get("1M")
    if hyg_1m is not None and lqd_1m is not None:
        diff = hyg_1m - lqd_1m
        if diff < -2:
            bullets.append(f'High-yield (HYG {fmt_pct(hyg_1m)}) is underperforming investment-grade bonds (LQD {fmt_pct(lqd_1m)}) by {_r(f"{abs(diff):.1f}pp")} — a divergence that signals rising credit stress and declining risk appetite.')
        elif diff > 2:
            bullets.append(f'High-yield (HYG {fmt_pct(hyg_1m)}) is outperforming investment-grade (LQD {fmt_pct(lqd_1m)}) — credit markets are risk-on; spreads are compressing, which is supportive for equities.')
        else:
            bullets.append(f'High-yield (HYG {fmt_pct(hyg_1m)}) and investment-grade bonds (LQD {fmt_pct(lqd_1m)}) are moving in tandem — no significant credit stress signal from the bond market.')

    # HY OAS level
    hy_cur = credit.get("hy", {}).get("current")
    if hy_cur is not None:
        if hy_cur > 7:
            bullets.append(f'HY OAS of {_r(f"{hy_cur:.2f}%")} is in stress territory (>7%). Corporate credit is pricing in meaningful default risk — a significant risk-off warning.')
        elif hy_cur > 4:
            bullets.append(f'HY OAS of {_w(f"{hy_cur:.2f}%")} is in the caution zone (4–7%). Spreads signal rising but not acute credit risk.')
        else:
            bullets.append(f'HY OAS of {_b(f"{hy_cur:.2f}%")} is tight (<4%) — credit is in risk-on mode with strong demand for yield.')

    return _interp_card("Analyst Read — Volatility & Credit", bullets)


def _interp_sectors(sectors: dict) -> str:
    bullets = []

    # Sort sectors by 1M
    ranked = sorted(
        [(ticker, SECTORS[ticker], sectors.get(ticker, {})) for ticker in SECTORS],
        key=lambda x: x[2].get("1M") or -999,
        reverse=True
    )

    top3 = [(t, n, d.get("1M")) for t, n, d in ranked[:3]  if d.get("1M") is not None]
    bot3 = [(t, n, d.get("1M")) for t, n, d in ranked[-3:] if d.get("1M") is not None]
    bot3.reverse()

    if top3:
        top_str = ", ".join(f'{_b(n)} ({fmt_pct(v)})' for t, n, v in top3)
        bullets.append(f'Leading sectors (1M): {top_str}.')
    if bot3:
        bot_str = ", ".join(f'{_r(n)} ({fmt_pct(v)})' for t, n, v in bot3)
        bullets.append(f'Lagging sectors (1M): {bot_str}.')

    # Leadership character
    defensive = {"XLU", "XLP", "XLRE", "XLV"}
    growth    = {"XLK", "XLC", "XLY"}
    cyclical  = {"XLF", "XLI", "XLB", "XLE"}
    top3_t = {t for t, n, v in top3}
    bot3_t = {t for t, n, v in bot3}

    if len(top3_t & defensive) >= 2:
        bullets.append(f'Defensive sectors are leading — Utilities, Staples, and/or Healthcare outperforming. This signals {_r("risk-off rotation")}: investors are preferring capital preservation over growth.')
    elif len(top3_t & growth) >= 2:
        bullets.append(f'Growth/tech sectors are leading — a {_b("risk-on signal")} consistent with strong earnings expectations and/or falling rate expectations.')
    elif len(top3_t & cyclical) >= 2:
        bullets.append(f'Cyclical sectors are leading (Financials, Industrials, Materials, Energy) — consistent with {_b("early-to-mid cycle dynamics")} or improving economic growth expectations.')

    if len(bot3_t & defensive) >= 2:
        bullets.append(f'Defensive sectors are lagging — investors are rotating {_b("out of safety")} into higher-beta names, consistent with improving risk appetite.')

    # Breadth: how many sectors are positive 1M
    pos_1m = sum(1 for _, _, d in ranked if (d.get("1M") or 0) > 0)
    if pos_1m >= 9:
        bullets.append(f'{_b(f"{pos_1m}/11")} sectors are positive over the past month — very broad participation, a constructive breadth signal.')
    elif pos_1m >= 6:
        bullets.append(f'{_w(f"{pos_1m}/11")} sectors are positive over the past month — moderate breadth; leadership is reasonably broad but not universal.')
    else:
        bullets.append(f'Only {_r(f"{pos_1m}/11")} sectors are positive over the past month — narrow breadth. Be cautious of index-level strength masking underlying weakness.')

    return _interp_card("Analyst Read — Sector Rotation", bullets)


def _interp_macro(buffett: dict, cape: dict, recession: dict, credit: dict,
                  money_mkt: dict, global_idx: dict) -> str:
    bullets = []

    # Combined valuation picture
    bi_val   = buffett.get("current")
    bi_label = buffett.get("label", "")
    cape_val = cape.get("current")
    cape_avg = cape.get("avg")

    if bi_val is not None and cape_val is not None:
        if bi_val > 150 and cape_val > 30:
            bullets.append(f'Both the Buffett Index ({_r(f"{bi_val:.0f}%")}) and Shiller CAPE ({_r(f"{cape_val:.1f}×")}) are in historically elevated territory. Dual confirmation of stretched valuations raises long-term risk of below-average returns — even if the market can remain elevated for an extended period.')
        elif bi_val > 100 or cape_val > 25:
            bullets.append(f'Buffett Index ({_w(f"{bi_val:.0f}%")}) and/or CAPE ({_w(f"{cape_val:.1f}×")}) signal above-average valuation. Elevated multiples compress prospective 10-year returns but do not dictate the near-term direction.')
        else:
            bullets.append(f'Buffett Index ({_b(f"{bi_val:.0f}%")}) and CAPE ({_b(f"{cape_val:.1f}×")}) are at or below historical averages — valuations are not a meaningful headwind here.')
    elif cape_val is not None and cape_avg is not None:
        premium = round((cape_val / cape_avg - 1) * 100)
        if premium > 40:
            bullets.append(f'Shiller CAPE of {_r(f"{cape_val:.1f}×")} sits {_r(f"{premium:.0f}%")} above its historical average ({cape_avg:.1f}×). At this premium, expected 10-year S&P 500 returns historically average 2–5% annually — well below the long-run mean.')
        elif premium > 15:
            bullets.append(f'CAPE of {_w(f"{cape_val:.1f}×")} is {_w(f"{premium:.0f}%")} above its historical average — elevated but not extreme. Future returns may be below-average without implying an imminent correction.')

    # Recession probability
    rec_val   = recession.get("current")
    rec_label = recession.get("label", "")
    if rec_val is not None:
        if rec_val >= 30:
            bullets.append(f'The NY Fed recession model shows {_r(f"{rec_val:.1f}%")} 12-month probability ({_r(rec_label)}). Historically, readings above 30% have been associated with elevated near-term economic risk. Note: the model is published with a ~3-month lag.')
        elif rec_val >= 15:
            bullets.append(f'Recession probability is {_w(f"{rec_val:.1f}%")} ({rec_label}) — the yield-curve model is flashing caution. Watch for confirmation in leading economic indicators (PMI, initial claims).')
        else:
            bullets.append(f'NY Fed recession probability is {_b(f"{rec_val:.1f}%")} ({rec_label}) — the yield-curve signal is benign, consistent with continued economic expansion.')

    # Credit spreads
    hy_cur = credit.get("hy", {}).get("current")
    ig_cur = credit.get("ig", {}).get("current")
    if hy_cur is not None:
        if hy_cur > 7:
            bullets.append(f'HY OAS of {_r(f"{hy_cur:.2f}%")} is in stress territory — corporate credit is pricing in meaningful default risk, a significant risk-off signal from the bond market.')
        elif hy_cur > 4:
            bullets.append(f'HY OAS of {_w(f"{hy_cur:.2f}%")} is in the caution zone (4–7%). Spreads are elevated above the tightest levels; credit risk is rising but not yet systemic.')
        else:
            bullets.append(f'HY OAS of {_b(f"{hy_cur:.2f}%")} is tight (<4%) — credit markets are firmly risk-on, with strong demand for yield and compressed default-risk premiums.')

    # Money markets
    mm_val = money_mkt.get("current")
    mm_chg = money_mkt.get("change_yoy")
    if mm_val is not None:
        if mm_chg is not None and mm_chg > 0.5:
            bullets.append(f'US money market assets stand at {_v(f"${mm_val:.2f}T")} (+${mm_chg:.2f}T YoY). Rising cash balances represent "dry powder" — historically, record MMF balances have been a contrarian positive once sentiment stabilises and cash rotates back into equities.')
        elif mm_chg is not None and mm_chg < -0.3:
            bullets.append(f'Money market assets ({_v(f"${mm_val:.2f}T")}, {fmt_pct(abs(mm_chg / mm_val * 100))} outflows YoY) are declining — cash is rotating back into risk assets, a constructive near-term signal for equities.')
        else:
            bullets.append(f'Money market assets are at {_v(f"${mm_val:.2f}T")} — broadly stable, reflecting neither an acute flight to safety nor a broad rush into risk.')

    # US vs International
    spy_ytd = global_idx.get("SPY", {}).get("YTD")
    efa_ytd = global_idx.get("EFA", {}).get("YTD")
    eem_ytd = global_idx.get("EEM", {}).get("YTD")
    if spy_ytd is not None and efa_ytd is not None:
        diff = spy_ytd - efa_ytd
        if diff > 5:
            bullets.append(f'US equities (SPY {fmt_pct(spy_ytd)} YTD) are significantly outperforming developed international markets (EFA {fmt_pct(efa_ytd)} YTD) — the US exceptionalism trade remains intact.')
        elif diff < -5:
            bullets.append(f'International developed markets (EFA {fmt_pct(efa_ytd)} YTD) are outperforming US equities (SPY {fmt_pct(spy_ytd)} YTD) by {abs(diff):.1f}pp — a notable rotation away from US assets, potentially driven by USD weakness or relative valuation.')
        if eem_ytd is not None and eem_ytd > spy_ytd + 5:
            bullets.append(f'Emerging markets (EEM {fmt_pct(eem_ytd)} YTD) are also outperforming US equities — a broad risk-on / global growth-positive signal.')

    return _interp_card("Analyst Read — Macro Signals", bullets)


def _interp_commodities(commodities: dict) -> str:
    bullets = []

    gld  = commodities.get("GLD",  {})
    slv  = commodities.get("SLV",  {})
    uso  = commodities.get("USO",  {})
    ung  = commodities.get("UNG",  {})
    cper = commodities.get("CPER", {})

    gld_1m  = gld.get("1M")
    gld_1y  = gld.get("1Y")
    uso_1m  = uso.get("1M")
    cper_1m = cper.get("1M")
    slv_1m  = slv.get("1M")

    # Gold
    if gld_1m is not None:
        if gld_1m > 5:
            bullets.append(f'Gold surged {_b(fmt_pct(gld_1m))} over the past month — a strong safe-haven / inflation-hedge signal. Sustained gold outperformance typically reflects USD weakness, geopolitical risk, and/or declining real interest rates.')
        elif gld_1m > 2:
            bullets.append(f'Gold gained {_b(fmt_pct(gld_1m))} over the past month — modest safe-haven demand with the precious metals complex in an uptrend.')
        elif gld_1m < -3:
            bullets.append(f'Gold fell {_r(fmt_pct(gld_1m))} over the past month — a risk-on environment where investors are reducing safe-haven exposure in favour of equities.')
        else:
            bullets.append(f'Gold returned {fmt_pct(gld_1m)} over the past month — broadly flat, with no strong directional safe-haven or risk-on signal.')

    if gld_1y is not None and gld_1y > 20:
        bullets.append(f'Gold\'s 1-year return of {_b(fmt_pct(gld_1y))} is exceptional. Sustained central bank buying, de-dollarisation trends, and/or persistent inflation expectations are likely drivers.')

    # Gold / Copper ratio (risk sentiment)
    if gld_1m is not None and cper_1m is not None:
        gc_diff = gld_1m - cper_1m
        if gc_diff > 5:
            bullets.append(f'Gold is outperforming copper by {_r(f"{gc_diff:.1f}pp")} (1M). A rising Gold/Copper ratio is a classic {_r("risk-off / recession-concern")} signal — gold\'s safe-haven premium is expanding relative to industrial demand.')
        elif gc_diff < -5:
            bullets.append(f'Copper is outperforming gold by {_b(f"{abs(gc_diff):.1f}pp")} (1M). A falling Gold/Copper ratio signals {_b("economic optimism")} — industrial demand is outpacing safe-haven demand.')

    # Oil
    if uso_1m is not None:
        if uso_1m > 5:
            bullets.append(f'Crude oil (USO) gained {_b(fmt_pct(uso_1m))} over the past month — consistent with supply tightness or recovering demand. Elevated oil adds inflationary pressure and can weigh on consumer and transport margins.')
        elif uso_1m < -5:
            bullets.append(f'Crude oil (USO) fell {_r(fmt_pct(uso_1m))} over the past month — disinflationary, which can support consumer spending and reduce margin pressure on energy-intensive businesses.')

    # Gold vs Silver (industrial vs pure safe haven)
    if gld_1m is not None and slv_1m is not None:
        gs_diff = slv_1m - gld_1m
        if gs_diff > 4:
            bullets.append(f'Silver is outperforming gold by {_b(f"{gs_diff:.1f}pp")} (1M) — when silver leads, it signals rising industrial activity alongside precious-metal demand, a broadly bullish macro read.')
        elif gs_diff < -4:
            bullets.append(f'Gold is outperforming silver by {abs(gs_diff):.1f}pp (1M) — indicating pure safe-haven demand rather than industrial growth. Silver\'s lag suggests muted near-term economic growth expectations.')

    # Overall commodity complex tone
    comm_vals = [commodities.get(t, {}).get("1M") for t in ["GLD", "SLV", "USO", "CPER"]]
    valid = [v for v in comm_vals if v is not None]
    if valid:
        avg_comm = sum(valid) / len(valid)
        if avg_comm > 3:
            bullets.append(f'The commodity complex is broadly rising (avg 1M: {_b(fmt_pct(avg_comm))}) — consistent with rising inflation expectations, strong global demand, or supply constraints.')
        elif avg_comm < -2:
            bullets.append(f'The commodity complex is broadly declining (avg 1M: {_r(fmt_pct(avg_comm))}) — disinflationary, often a signal of weakening global demand or easing supply bottlenecks.')

    return _interp_card("Analyst Read — Commodities", bullets)


def _perf_from_hist(close, n_days):
    if len(close) <= n_days:
        return None
    return pct_chg(float(close.iloc[-1]), float(close.iloc[-(n_days + 1)]))

def _ytd_perf(hist):
    this_year = datetime.date.today().year
    ytd_idx = next((i for i, d in enumerate(hist.index) if d.year >= this_year), None)
    if ytd_idx is None or ytd_idx == 0:
        return None
    return pct_chg(float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[ytd_idx - 1]))

# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch_treasury_yields() -> dict:
    """
    Full yield curve from the US Treasury XML API (no API key needed).

    The endpoint paginates 300 rows per page, oldest-first.  As of March 2026
    the dataset spans pages 0-30 (page 30 = most recent 50 entries).  To get
    ~14 months of daily data we fetch a small window of pages ending at the
    last non-empty page, discovered by probing downward from an upper bound.

    Switched from FRED CSV (which hangs due to Cloudflare bot-detection) to:
      home.treasury.gov/resource-center/data-chart-center/interest-rates/
      pages/xml?data=daily_treasury_yield_curve&field_tdate_value=202603&page=N
    """
    import re

    result   = {"current": {}, "history": []}
    today    = datetime.date.today()
    cutoff   = (today - datetime.timedelta(days=400)).isoformat()
    date_map = {}

    _BASE = ("https://home.treasury.gov/resource-center/data-chart-center/"
             "interest-rates/pages/xml"
             "?data=daily_treasury_yield_curve&field_tdate_value=202603&page=")
    _HDR  = {"User-Agent": "Mozilla/5.0 (compatible; ValuationSuite/2.0)"}

    def _fetch_page(page: int) -> list:
        """Return list of {date, maturity: val} dicts for one page."""
        rows = []
        try:
            req = Request(_BASE + str(page), headers=_HDR)
            with urlopen(req, timeout=25) as r:
                txt = r.read().decode("utf-8", errors="replace")
            for entry in re.findall(r"<m:properties>(.*?)</m:properties>",
                                    txt, re.DOTALL):
                dm = re.search(r"<d:NEW_DATE[^>]*>(\d{4}-\d{2}-\d{2})", entry)
                if not dm:
                    continue
                date_str = dm.group(1)
                rec = {"date": date_str}
                for label, xml_field in TREASURY_XML_MAP.items():
                    vm = re.search(rf"<d:{xml_field}[^>]*>([0-9.]+)", entry)
                    if vm:
                        rec[label] = float(vm.group(1))
                rows.append(rec)
        except Exception:
            pass
        return rows

    # Probe to find the last non-empty page (binary-search style, upper bound 60)
    # Dataset grows by ~1 row/trading day; page size = 300 rows.
    # Upper bound of 60 is safe until ~2036.
    lo, hi = 0, 60
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fetch_page(mid):
            lo = mid
        else:
            hi = mid - 1
    last_page = lo

    # Fetch last_page and enough prior pages to cover cutoff (~2 pages = ~550 rows)
    pages_to_fetch = list(range(max(0, last_page - 2), last_page + 1))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for page_rows in pool.map(_fetch_page, pages_to_fetch):
            for rec in page_rows:
                if rec["date"] >= cutoff:
                    date_map[rec["date"]] = rec

    if not date_map:
        return result

    all_rows = sorted(date_map.values(), key=lambda x: x["date"])
    result["current"] = all_rows[-1]
    result["history"] = all_rows
    return result


def fetch_vix() -> dict:
    """VIX + VVIX from yfinance."""
    result = {"current": None, "avg_1y": None, "dates": [], "values": [], "vvix": None}
    try:
        h = yf.Ticker("^VIX").history(period="1y")
        if not h.empty:
            result["current"]  = round(float(h["Close"].iloc[-1]), 2)
            result["avg_1y"]   = round(float(h["Close"].mean()), 2)
            result["dates"]    = [str(d.date()) for d in h.index]
            result["values"]   = [round(float(v), 2) for v in h["Close"]]
    except Exception:
        pass
    try:
        h2 = yf.Ticker("^VVIX").history(period="5d")
        if not h2.empty:
            result["vvix"] = round(float(h2["Close"].iloc[-1]), 1)
    except Exception:
        pass
    return result


def fetch_cross_asset() -> dict:
    """Price history + perf calculations for every cross-asset ticker."""
    tickers = [t for t, _, _ in CROSS_ASSET]
    result  = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            return ticker, {
                "price":  round(price, 4),
                "1D":     _perf_from_hist(close, 1),
                "1W":     _perf_from_hist(close, 5),
                "1M":     _perf_from_hist(close, 21),
                "3M":     _perf_from_hist(close, 63),
                "6M":     _perf_from_hist(close, 126),
                "YTD":    _ytd_perf(h),
                "vs_200": pct_chg(price, ma200) if ma200 else None,
                "dates":  [str(d.date()) for d in h.index[-126:]],
                "closes": [round(float(v), 4) for v in close.iloc[-126:]],
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


def fetch_sectors() -> dict:
    """Sector ETF performance."""
    tickers = list(SECTORS.keys())
    result  = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            return ticker, {
                "price": round(price, 2),
                "1D":    _perf_from_hist(close, 1),
                "1W":    _perf_from_hist(close, 5),
                "1M":    _perf_from_hist(close, 21),
                "3M":    _perf_from_hist(close, 63),
                "YTD":   _ytd_perf(h),
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


def fetch_yf_yield_history() -> dict:
    """Historical yields from yfinance for 3M, 5Y, 10Y, 30Y."""
    result = {}
    def _one(args):
        label, ticker = args
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if not h.empty:
                return label, {
                    "dates":   [str(d.date()) for d in h.index],
                    "values":  [round(float(v), 3) for v in h["Close"]],
                    "current": round(float(h["Close"].iloc[-1]), 3),
                }
        except Exception:
            pass
        return label, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for label, data in pool.map(_one, YF_YIELDS.items()):
            if data:
                result[label] = data
    return result


def fetch_buffett_index() -> dict:
    """
    Buffett Indicator: Wilshire 5000 (^W5000 via yfinance) / US Nominal GDP.
    Wilshire 5000 index points ≈ total US market cap in billions USD.
    GDP from Federal Reserve Z.1 f2.csv — FA086902005.Q (millions, SAAR).
    Converted to billions (÷1000) to match traditional Buffett calculation.
    Sampled quarterly; 10 years of history.
    Interpretation: <80 undervalued · 80-100 fair · 100-120 modestly overvalued
                    120-150 overvalued · >150 significantly overvalued
    """
    import re as _re, zipfile as _zf, io as _io, tempfile as _tmp, time as _tm
    import requests as _req

    result = {"current": None, "label": None, "color": None, "history": []}
    try:
        # ── Wilshire 5000 via yfinance (10y daily → quarterly last close) ──────
        w_hist = yf.Ticker("^W5000").history(period="10y")
        if w_hist.empty:
            return result
        if hasattr(w_hist.index, "tz") and w_hist.index.tz is not None:
            w_hist.index = w_hist.index.tz_convert(None)
        w_quarterly = {}   # "YYYY-Qn" → (date_str, close_val)
        for dt, row in w_hist.iterrows():
            q_key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            w_quarterly[q_key] = (str(dt.date()), float(row["Close"]))

        # ── GDP from Federal Reserve Z.1 (f2.csv, FA086902005.Q, millions SAAR) ─
        # Reuse the same Z.1 ZIP already fetched/cached for money markets
        page = _req.get("https://www.federalreserve.gov/releases/z1/",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = _re.search(r'href=["\'](/releases/z1/\d{8}/z1_csv_files\.zip)["\']',
                       page.text)
        zip_url = ("https://www.federalreserve.gov" + m.group(1)) if m else \
                  "https://www.federalreserve.gov/releases/z1/20260109/z1_csv_files.zip"

        cache_path = os.path.join(
            _tmp.gettempdir(),
            "z1_csv_" + zip_url.split("/")[-2] + ".zip"
        )
        if os.path.exists(cache_path) and (_tm.time() - os.path.getmtime(cache_path)) < 86400:
            with open(cache_path, "rb") as _f:
                z_bytes = _f.read()
        else:
            resp = _req.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            z_bytes = resp.content
            with open(cache_path, "wb") as _f:
                _f.write(z_bytes)

        zf = _zf.ZipFile(_io.BytesIO(z_bytes))
        content = zf.read("csv/f2.csv").decode("utf-8", errors="replace")

        # Parse FA086902005.Q (GDP in millions SAAR) → billions ÷ 1000
        gdp_data = {}
        lines = content.strip().split("\n")
        header = lines[0].split(",")
        gdp_col = next((i for i, h in enumerate(header) if "FA086902005" in h), None)
        if gdp_col is None:
            return result
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= gdp_col:
                continue
            date_str = parts[0].strip()   # e.g. "2024:Q3"
            val_str  = parts[gdp_col].strip()
            if ":Q" not in date_str:
                continue
            year_s, q_s = date_str.split(":Q")
            q_end = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(q_s)
            if not q_end:
                continue
            fv = _safe(val_str)
            if fv is not None and fv > 0:
                gdp_data[f"{year_s}-{q_end}"] = fv / 1000   # convert millions → billions

        if not gdp_data:
            return result

        # Forward-fill quarterly GDP to each Wilshire quarterly date
        gdp_sorted = sorted(gdp_data.keys())
        def _latest_gdp(date_str):
            best = None
            for gd in gdp_sorted:
                if gd <= date_str:
                    best = gd
                else:
                    break
            return gdp_data.get(best) if best else None

        # ── Build ratio history ────────────────────────────────────────────────
        history = []
        for q_key in sorted(w_quarterly.keys()):
            date_str, w_val = w_quarterly[q_key]
            gdp_val = _latest_gdp(date_str)
            if gdp_val and gdp_val > 0 and w_val > 0:
                ratio = round(w_val / gdp_val * 100, 1)
                history.append({"date": date_str, "ratio": ratio})

        if not history:
            return result

        current = history[-1]["ratio"]
        if   current < 80:  label, color = "Undervalued",              "#00c896"
        elif current < 100: label, color = "Fair Value",               "#4fc08d"
        elif current < 120: label, color = "Modestly Overvalued",      "#f0a500"
        elif current < 150: label, color = "Overvalued",               "#e07b39"
        else:               label, color = "Significantly Overvalued", "#e05c5c"

        result.update({"current": current, "label": label, "color": color, "history": history})
    except Exception:
        pass
    return result


def fetch_money_markets() -> dict:
    """
    Total US money market fund assets — FL634090005.Q from Federal Reserve Z.1
    (Table L.121, Money Market Funds, total financial assets, millions of dollars).
    Source: federalreserve.gov/releases/z1 CSV ZIP (direct download, no API key required).
    The ZIP is cached for 24 hours in the system temp directory.
    """
    import re as _re, zipfile as _zf, io as _io, tempfile as _tmp, time as _tm
    import requests as _req

    result = {"current": None, "change_yoy": None, "history": []}
    cutoff_year = (datetime.date.today() - datetime.timedelta(days=365 * 7)).year

    try:
        # ── 1. Find the latest Z.1 release ZIP URL ───────────────────────────────
        page = _req.get("https://www.federalreserve.gov/releases/z1/",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = _re.search(r'href=["\'](/releases/z1/\d{8}/z1_csv_files\.zip)["\']',
                       page.text)
        zip_url = ("https://www.federalreserve.gov" + m.group(1)) if m else \
                  "https://www.federalreserve.gov/releases/z1/20260109/z1_csv_files.zip"

        # ── 2. Cache the ZIP for 24 h to avoid re-downloading 7 MB each run ─────
        cache_path = os.path.join(
            _tmp.gettempdir(),
            "z1_csv_" + zip_url.split("/")[-2] + ".zip"
        )
        if os.path.exists(cache_path) and (_tm.time() - os.path.getmtime(cache_path)) < 86400:
            with open(cache_path, "rb") as _f:
                z_bytes = _f.read()
        else:
            resp = _req.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            z_bytes = resp.content
            with open(cache_path, "wb") as _f:
                _f.write(z_bytes)

        # ── 3. Read L.121 CSV — FL634090005.Q = Money Market Funds total assets ──
        zf      = _zf.ZipFile(_io.BytesIO(z_bytes))
        content = zf.read("csv/l121.csv").decode("utf-8", errors="replace")

        rows = []
        for line in content.strip().split("\n"):
            if not line or line.startswith("date"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            date_str = parts[0].strip()   # e.g. "2024:Q3"
            val_str  = parts[1].strip()   # FL634090005.Q (millions)
            if ":Q" not in date_str:
                continue
            year_s, q_s = date_str.split(":Q")
            if int(year_s) < cutoff_year:
                continue
            q_end = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(q_s)
            if not q_end:
                continue
            fv = _safe(val_str)
            if fv is not None and fv > 0:
                rows.append({"date": f"{year_s}-{q_end}", "assets": round(fv / 1_000_000, 2)})

        if rows:
            rows.sort(key=lambda x: x["date"])
            result["history"]    = rows
            result["current"]    = rows[-1]["assets"]
            if len(rows) >= 5:
                result["change_yoy"] = round(result["current"] - rows[-5]["assets"], 2)
    except Exception:
        pass
    return result


def fetch_global_equity() -> dict:
    """
    SPY vs EFA (developed ex-US) vs EEM (emerging markets) — 1-year price history.
    Used for US vs international equity rotation analysis.
    """
    tickers = ["SPY", "EFA", "EEM"]
    result  = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            return ticker, {
                "price":  round(price, 2),
                "1W":     _perf_from_hist(close, 5),
                "1M":     _perf_from_hist(close, 21),
                "3M":     _perf_from_hist(close, 63),
                "6M":     _perf_from_hist(close, 126),
                "YTD":    _ytd_perf(h),
                "dates":  [str(d.date()) for d in h.index],
                "closes": [round(float(v), 4) for v in close],
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=3) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


def fetch_shiller_cape() -> dict:
    """
    Shiller CAPE (Cyclically Adjusted P/E) ratio.
    Primary source: multpl.com monthly table (free, no API key).
    Fallback: yfinance ^GSPC trailing P/E (TTM, not 10-yr avg).
    Historical avg ~17; >30 = expensive; >40 = extreme.
    """
    import re as _re
    result = {"current": None, "avg": None, "label": None, "color": None, "history": []}
    try:
        url = "https://www.multpl.com/shiller-pe/table/by-month"
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        with urlopen(req, timeout=20) as r:
            html = r.read().decode("utf-8", errors="replace")

        # Each data row: <td>Feb 1, 2026</td><td>\n&#x2002;\n40.00\n</td>
        row_re = _re.compile(
            r'<td>([A-Za-z]+ \d{1,2},\s*\d{4})</td>\s*'
            r'<td[^>]*>[\s\S]*?(\d+\.?\d*)\s*</td>'
        )
        rows = []
        for m in row_re.finditer(html):
            try:
                dt = datetime.datetime.strptime(m.group(1).strip(), "%b %d, %Y")
                date_iso = str(dt.date())
                val = float(m.group(2))
                if date_iso >= "1990-01-01":
                    rows.append({"date": date_iso, "cape": round(val, 2)})
            except Exception:
                continue

        if rows:
            rows.sort(key=lambda x: x["date"])
            # Monthly data — sample every other point to keep chart manageable
            result["history"] = rows[::2]
            result["current"] = rows[-1]["cape"]
            # Use full scraped period avg (since 1990)
            result["avg"] = round(sum(r["cape"] for r in rows) / len(rows), 1)
    except Exception:
        pass

    # Fallback: yfinance trailing P/E (TTM, approximate — not 10-yr smoothed)
    if result["current"] is None:
        try:
            info = yf.Ticker("^GSPC").fast_info
            pe = getattr(info, "pe_ratio", None) or _safe(
                yf.Ticker("SPY").info.get("trailingPE"))
            if pe:
                result["current"] = round(float(pe), 1)
                result["avg"]     = 17.0  # long-run historical mean
        except Exception:
            pass

    cur = result.get("current")
    if cur is not None:
        if   cur < 15: result["label"], result["color"] = "Undervalued", "#00c896"
        elif cur < 22: result["label"], result["color"] = "Fair Value",  "#4fc08d"
        elif cur < 30: result["label"], result["color"] = "Elevated",    "#f0a500"
        elif cur < 40: result["label"], result["color"] = "Expensive",   "#e07b39"
        else:          result["label"], result["color"] = "Extreme",     "#e05c5c"
    return result


def fetch_recession_prob() -> dict:
    """
    NY Fed 12-month US recession probability computed from the yield-curve probit model
    (Estrella & Mishkin).  P = Φ(−0.5261 + −0.5146 × spread) where spread = 10Y − 3M.
    Source data: yfinance ^TNX (10Y) and ^IRX (3M), monthly resampled.
    This replicates FRED RECPROUSM156N without requiring FRED access.
    >30% = high risk; >50% = danger zone.
    """
    import pandas as _pd, numpy as _np
    try:
        from scipy.stats import norm as _norm
    except ImportError:
        return {"current": None, "label": None, "color": None, "history": []}

    result = {"current": None, "label": None, "color": None, "history": []}
    try:
        _A, _B = -0.5261, -0.5146   # probit coefficients

        tnx = yf.Ticker("^TNX").history(period="25y")["Close"]
        irx = yf.Ticker("^IRX").history(period="25y")["Close"]
        tnx.index = tnx.index.tz_convert("UTC").normalize()
        irx.index = irx.index.tz_convert("UTC").normalize()

        spread = tnx.resample("ME").last() - irx.resample("ME").last()
        spread = spread.dropna()
        prob   = _pd.Series(
            [_norm.cdf(_A + _B * s) * 100 for s in spread.values],
            index=spread.index,
        )
        prob = prob[prob.index >= "2000-01-01"].dropna()

        rows = [{"date": str(d.date()), "prob": round(float(v), 2)}
                for d, v in zip(prob.index, prob.values)]
        if rows:
            result["history"] = rows
            cur = rows[-1]["prob"]
            result["current"] = cur
            if   cur < 10: result["label"], result["color"] = "Low",         "#00c896"
            elif cur < 20: result["label"], result["color"] = "Modest",      "#4fc08d"
            elif cur < 30: result["label"], result["color"] = "Elevated",    "#f0a500"
            elif cur < 50: result["label"], result["color"] = "High Risk",   "#e07b39"
            else:          result["label"], result["color"] = "Danger Zone", "#e05c5c"
    except Exception:
        pass
    return result


def fetch_credit_spreads() -> dict:
    """
    Credit spread proxies computed from yfinance ETF data (FRED is inaccessible).

    HY proxy: HYG (iShares iBoxx $ High Yield) forward yield − 5Y Treasury (^FVX).
    IG proxy: IGSB (iShares 1-5Y IG Corporate) forward yield − 3M Treasury (^IRX).

    Forward yield = most recent monthly dividend × 12 / current price.
    History uses rolling monthly forward yields; clipped to ≥ 0 and ≤ 20% (HY) / 8% (IG)
    to suppress dividend-timing artefacts from double- or missed-payment months.
    """
    import pandas as _pd, numpy as _np

    result = {"hy": {"current": None, "history": []},
              "ig": {"current": None, "history": []}}

    try:
        # ── Treasury benchmarks ───────────────────────────────────────────────────
        fvx_info = yf.Ticker("^FVX").fast_info
        irx_info = yf.Ticker("^IRX").fast_info
        tsy5  = float(fvx_info.last_price)   # 5Y Treasury (%)
        tsy3m = float(irx_info.last_price)   # 3M Treasury (%)

        def _fwd_spread_series(ticker, benchmark_series, lo_clip, hi_clip):
            """Rolling-12M trailing yield spread vs benchmark — stable across div timing."""
            h = yf.Ticker(ticker).history(period="6y")[["Close", "Dividends"]]
            h.index = h.index.tz_convert("UTC").normalize()
            bm = yf.Ticker(benchmark_series).history(period="6y")[["Close"]]
            bm.index = bm.index.tz_convert("UTC").normalize()

            # 63-day (~3 month) rolling sum, annualised ×4.
            # Shorter window keeps the yield responsive during rapid rate moves
            # (252-day window lagged Fed hikes by a full year, causing IG to read ~0%).
            h["roll_div"]    = h["Dividends"].rolling(63, min_periods=45).sum()
            h["trail_yield"] = h["roll_div"] / h["Close"] * 100 * 4   # annualise
            # Align benchmark to ETF trading days
            bm_aligned = bm["Close"].reindex(h.index, method="ffill")
            spread_d = (h["trail_yield"] - bm_aligned).dropna()
            spread_d = spread_d.clip(lower=lo_clip, upper=hi_clip)
            # Monthly sample (last business day each month)
            spread_m = spread_d.resample("ME").last()
            spread_m = spread_m[spread_m.index >= "2020-01-01"]

            rows = [{"date": str(d.date()), "spread": round(float(v), 3)}
                    for d, v in zip(spread_m.index, spread_m.values)]
            cur = rows[-1]["spread"] if rows else None
            return cur, rows   # monthly cadence

        # ── HY: HYG − ^FVX ────────────────────────────────────────────────────────
        def _current_fwd_yield(ticker):
            h = yf.Ticker(ticker).history(period="35d")
            divs = h[h["Dividends"] > 0]["Dividends"]
            if divs.empty:
                return None
            return float(divs.iloc[-1]) * 12 / float(h["Close"].iloc[-1]) * 100

        hy_yield = _current_fwd_yield("HYG")
        ig_yield = _current_fwd_yield("IGSB")
        hy_cur = round(hy_yield - tsy5,  3) if hy_yield else None
        ig_cur = round(ig_yield - tsy3m, 3) if ig_yield else None

        hy_hist_cur, hy_hist = _fwd_spread_series("HYG",  "^FVX",  0.0, 20.0)
        ig_hist_cur, ig_hist = _fwd_spread_series("IGSB", "^IRX",  0.0,  8.0)

        result["hy"]["current"] = hy_cur if hy_cur is not None else hy_hist_cur
        result["hy"]["history"] = hy_hist
        result["ig"]["current"] = ig_cur if ig_cur is not None else ig_hist_cur
        result["ig"]["history"] = ig_hist
    except Exception:
        pass
    return result


def fetch_cap_size() -> dict:
    """Cap size (SPY/MDY/IWM/IWC) and style (growth/value) ETF performance."""
    tickers = list(dict.fromkeys(
        [t for t, _, _ in CAP_SIZE] + [t for t, _, _ in CAP_STYLE]
    ))
    result = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            return ticker, {
                "price":  round(price, 2),
                "1D":     _perf_from_hist(close, 1),
                "1W":     _perf_from_hist(close, 5),
                "1M":     _perf_from_hist(close, 21),
                "3M":     _perf_from_hist(close, 63),
                "6M":     _perf_from_hist(close, 126),
                "YTD":    _ytd_perf(h),
                "dates":  [str(d.date()) for d in h.index],
                "closes": [round(float(v), 4) for v in close],
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


def fetch_global_indices() -> dict:
    """Comprehensive global market indices ETF performance."""
    tickers = list(dict.fromkeys([t for t, _, _ in GLOBAL_INDICES]))
    result = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="1y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            return ticker, {
                "price":  round(price, 2),
                "1D":     _perf_from_hist(close, 1),
                "1W":     _perf_from_hist(close, 5),
                "1M":     _perf_from_hist(close, 21),
                "3M":     _perf_from_hist(close, 63),
                "6M":     _perf_from_hist(close, 126),
                "YTD":    _ytd_perf(h),
                "dates":  [str(d.date()) for d in h.index],
                "closes": [round(float(v), 4) for v in close],
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


def fetch_commodities() -> dict:
    """Commodity ETF price history for the Commodities tab."""
    tickers = [t for t, _, _ in COMMODITIES]
    result  = {}

    def _one(ticker):
        try:
            h = yf.Ticker(ticker).history(period="2y")
            if h.empty:
                return ticker, None
            close = h["Close"]
            price = float(close.iloc[-1])
            return ticker, {
                "price":  round(price, 2),
                "1D":     _perf_from_hist(close, 1),
                "1W":     _perf_from_hist(close, 5),
                "1M":     _perf_from_hist(close, 21),
                "3M":     _perf_from_hist(close, 63),
                "6M":     _perf_from_hist(close, 126),
                "1Y":     _perf_from_hist(close, 252),
                "YTD":    _ytd_perf(h),
                "dates":  [str(d.date()) for d in h.index],
                "closes": [round(float(v), 4) for v in close],
            }
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for ticker, data in pool.map(_one, tickers):
            if data:
                result[ticker] = data
    return result


# ── Regime Score ───────────────────────────────────────────────────────────────
def compute_regime(cross: dict, vix: dict, treasury: dict) -> dict:
    """
    Five components × 20 pts each → 0-100 composite.
    Higher = more risk-on / bullish macro backdrop.
    """
    comps = {}
    total = 0
    available = 0

    # 1. Market momentum — SPY vs 200-day MA
    vs200 = cross.get("SPY", {}).get("vs_200")
    if vs200 is not None:
        s = max(0.0, min(20.0, 10.0 + vs200 * 2.0))
        comps["momentum"] = {
            "label": "Market Momentum",
            "score": round(s, 1),
            "detail": f"SPY {fmt_pct(vs200)} vs 200-day MA",
        }
        total += s; available += 1
    else:
        comps["momentum"] = {"label": "Market Momentum", "score": None, "detail": "No data"}

    # 2. Volatility — VIX level
    vix_val = vix.get("current")
    if vix_val is not None:
        s = 20 if vix_val < 15 else 16 if vix_val < 20 else 10 if vix_val < 25 else 5 if vix_val < 30 else 0
        comps["volatility"] = {
            "label": "Volatility (VIX)",
            "score": float(s),
            "detail": f"VIX {vix_val:.1f}  ·  1Y avg {vix.get('avg_1y', 0):.1f}",
        }
        total += s; available += 1
    else:
        comps["volatility"] = {"label": "Volatility (VIX)", "score": None, "detail": "No data"}

    # 3. Yield curve — 10Y minus 2Y
    cur = treasury.get("current", {})
    y10, y2 = cur.get("10Y"), cur.get("2Y")
    if y10 and y2:
        spread = y10 - y2
        s = 20 if spread > 1.0 else 16 if spread > 0.25 else 10 if spread > -0.25 else 5 if spread > -0.75 else 0
        comps["curve"] = {
            "label": "Yield Curve (10Y-2Y)",
            "score": float(s),
            "detail": f"Spread {spread:+.2f}%  ·  10Y {y10:.2f}%  ·  2Y {y2:.2f}%",
        }
        total += s; available += 1
    else:
        comps["curve"] = {"label": "Yield Curve (10Y-2Y)", "score": None, "detail": "No data"}

    # 4. Risk appetite — SPY 1M vs composite safe haven (TLT + GLD, equal weight)
    #    Gold is a co-equal safe-haven; ignoring it misses major risk-off episodes.
    spy_1m = cross.get("SPY", {}).get("1M")
    tlt_1m = cross.get("TLT", {}).get("1M")
    gld_1m = cross.get("GLD", {}).get("1M")
    sh_parts = [v for v in [tlt_1m, gld_1m] if v is not None]
    if spy_1m is not None and sh_parts:
        sh_avg = sum(sh_parts) / len(sh_parts)   # composite safe-haven return
        diff   = spy_1m - sh_avg                  # positive = equities leading = risk-on
        s = 20 if diff > 5 else 15 if diff > 1 else 10 if diff > -1 else 5 if diff > -5 else 0
        tlt_str = fmt_pct(tlt_1m) if tlt_1m is not None else "—"
        gld_str = fmt_pct(gld_1m) if gld_1m is not None else "—"
        comps["safe_haven"] = {
            "label": "Risk Appetite",
            "score": float(s),
            "detail": f"SPY {fmt_pct(spy_1m)} · TLT {tlt_str} · GLD {gld_str} (1M)",
        }
        total += s; available += 1
    else:
        comps["safe_haven"] = {"label": "Risk Appetite", "score": None, "detail": "No data"}

    # 5. Breadth proxy — IWM 3M vs SPY 3M (small-cap participation)
    iwm_3m = cross.get("IWM", {}).get("3M")
    spy_3m = cross.get("SPY", {}).get("3M")
    if iwm_3m is not None and spy_3m is not None:
        diff = iwm_3m - spy_3m
        s = 20 if diff > 3 else 15 if diff > 0 else 8 if diff > -3 else 4 if diff > -6 else 0
        comps["breadth"] = {
            "label": "Breadth (IWM vs SPY)",
            "score": float(s),
            "detail": f"IWM {fmt_pct(iwm_3m)} vs SPY {fmt_pct(spy_3m)} past 3 months",
        }
        total += s; available += 1
    else:
        comps["breadth"] = {"label": "Breadth (IWM vs SPY)", "score": None, "detail": "No data"}

    # Normalise if some components missing
    if 0 < available < 5:
        total = total * 5 / available
    score = round(total)

    if score >= 80:    label, color = "Risk-On",            "#00c896"
    elif score >= 62:  label, color = "Moderately Bullish", "#4fc08d"
    elif score >= 42:  label, color = "Neutral",            "#f0a500"
    elif score >= 25:  label, color = "Cautious",           "#e07b39"
    else:              label, color = "Risk-Off",           "#e05c5c"

    return {"score": score, "label": label, "color": color, "components": comps}


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f14;--surface:#14171f;--sf2:#1c2030;--border:#252a3a;
  --text:#e8eaf2;--muted:#6b7194;--accent:#4f8ef7;
  --up:#00c896;--down:#e05c5c;--warn:#f0a500;--r:8px}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.5}
/* ── Header ── */
.hdr{background:var(--surface);border-bottom:1px solid var(--border);
  padding:16px 24px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.hdr-title{font-size:20px;font-weight:700}
.hdr-sub{font-size:12px;color:var(--muted)}
.hdr-badge{font-size:11px;background:var(--sf2);border:1px solid var(--border);
  border-radius:20px;padding:4px 12px;color:var(--muted);margin-left:auto}
/* ── Tabs ── */
.tab-bar{display:flex;background:var(--surface);border-bottom:2px solid var(--border);
  position:sticky;top:0;z-index:20;overflow-x:auto}
.tab-btn{background:none;border:none;border-bottom:3px solid transparent;
  padding:14px 22px;color:var(--muted);cursor:pointer;font-size:13px;
  font-weight:600;white-space:nowrap;transition:color .15s,border-color .15s}
.tab-btn.active,.tab-btn:hover{color:var(--accent);border-bottom-color:var(--accent)}
.tab-panel{display:none}.tab-panel.active{display:block}
/* ── Layout ── */
.container{max-width:1600px;margin:0 auto;padding:24px 20px 60px}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:20px;margin-bottom:20px}
.card-title{font-size:11px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:16px}
.sec-hdr{font-size:16px;font-weight:700;margin:28px 0 14px;
  padding-bottom:8px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:6px}
/* ── Grids ── */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.stat-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px;text-align:center}
.stat-val{font-size:26px;font-weight:700;line-height:1.1}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.04em}
.stat-sub{font-size:11px;color:var(--muted);margin-top:2px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
@media(max-width:900px){.two-col,.three-col{grid-template-columns:1fr}}
/* ── Tables ── */
.tbl-wrap{overflow-x:auto;border-radius:var(--r);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--sf2);color:var(--muted);font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.06em;padding:9px 12px;
  text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--sf2)}
td{padding:9px 12px;vertical-align:middle}
td.num{text-align:right;font-family:monospace}
th.num{text-align:right}
td.sym{font-weight:700;font-size:13px;color:var(--accent)}
/* ── Charts ── */
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px}
.chart-card h3{font-size:12px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:16px}
.chart-wrap{position:relative}
/* ── Regime gauge ── */
.regime-wrap{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
.regime-gauge{flex:0 0 280px;text-align:center}
.regime-gauge canvas{max-width:280px}
.regime-score-lbl{font-size:32px;font-weight:700;margin-top:-60px;line-height:1}
.regime-label{font-size:14px;font-weight:600;margin-top:6px}
.regime-comps{flex:1;min-width:260px}
.comp-row{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.comp-row:last-child{border-bottom:none}
.comp-bar-bg{flex:1;height:8px;background:var(--sf2);border-radius:4px;overflow:hidden}
.comp-bar{height:100%;border-radius:4px;transition:width .4s}
.comp-score{font-size:13px;font-weight:700;min-width:36px;text-align:right}
.comp-lbl{font-size:12px;width:220px;flex:0 0 220px;overflow:hidden}
.comp-detail{font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* ── Sector heatmap ── */
.sector-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.sector-cell{border-radius:var(--r);padding:14px 12px;text-align:center;border:1px solid var(--border)}
.sector-sym{font-size:13px;font-weight:700;margin-bottom:2px}
.sector-name{font-size:10px;color:var(--muted);margin-bottom:8px}
.sector-perf{font-size:15px;font-weight:700}
.sector-sub{font-size:10px;color:var(--muted);margin-top:2px}
/* ── Misc ── */
.up{color:var(--up)}.down{color:var(--down)}.warn{color:var(--warn)}.muted{color:var(--muted)}
.note{font-size:11px;color:var(--muted);font-style:italic}
.pill{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700}
/* ── Help button ── */
.help-btn{background:none;border:none;cursor:pointer;font-size:13px;
  color:var(--muted);padding:0 0 0 5px;line-height:1;vertical-align:middle;
  opacity:.65;transition:opacity .15s,color .15s}
.help-btn:hover{opacity:1;color:var(--accent)}
.card-title{display:flex;align-items:center;gap:0}
h3{display:flex;align-items:center;gap:0}
/* ── Help popover ── */
#help-pop{display:none;position:fixed;z-index:9999;
  background:var(--surface);border:1px solid var(--accent);
  border-radius:var(--r);padding:18px 18px 14px;
  max-width:420px;width:min(420px,92vw);
  box-shadow:0 8px 40px rgba(0,0,0,.7);
  font-size:12.5px;line-height:1.6;color:var(--text)}
#help-pop-title{font-weight:700;font-size:14px;color:var(--accent);
  margin-bottom:10px;padding-bottom:8px;
  border-bottom:1px solid var(--border)}
#help-pop-body b{color:var(--text)}
#help-pop-close{position:absolute;top:10px;right:12px;
  background:none;border:none;color:var(--muted);
  font-size:20px;line-height:1;cursor:pointer;padding:0}
#help-pop-close:hover{color:var(--text)}
/* ── Interpretation card ── */
.interp-card{background:var(--sf2);border:1px solid var(--border);
  border-left:4px solid var(--accent);border-radius:var(--r);
  padding:20px 24px;margin-top:24px;margin-bottom:4px}
.interp-card h4{font-size:12px;font-weight:700;color:var(--accent);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.interp-list{list-style:none;padding:0;margin:0}
.interp-list li{padding:6px 0 6px 18px;position:relative;
  font-size:13px;color:var(--text);border-bottom:1px solid var(--border);line-height:1.55}
.interp-list li:last-child{border-bottom:none}
.interp-list li::before{content:'›';position:absolute;left:0;
  color:var(--accent);font-weight:700}
"""

# ── HTML builder ───────────────────────────────────────────────────────────────
def _tab_overview(regime: dict, cross: dict) -> str:
    h = ""

    # ── Regime gauge + components ──────────────────────────────────────────────
    h += '<div class="card">'
    h += f'<div class="card-title">Macro Regime Score {help_btn("regime-score")}</div>'
    h += '<div class="regime-wrap">'

    # Gauge (half-doughnut via Chart.js)
    score = regime["score"]
    color = regime["color"]
    h += '<div class="regime-gauge">'
    h += '<canvas id="gauge-chart" height="160"></canvas>'
    h += f'<div class="regime-score-lbl" style="color:{color}">{score}</div>'
    h += f'<div class="regime-label" style="color:{color}">{regime["label"]}</div>'
    h += '<div class="note" style="margin-top:6px">0 = Risk-Off · 100 = Risk-On</div>'
    h += '</div>'

    # Component breakdown
    h += '<div class="regime-comps">'
    for key, c in regime["components"].items():
        s = c["score"]
        bar_w  = f"{(s / 20 * 100):.0f}%" if s is not None else "0%"
        bar_c  = "#00c896" if (s and s >= 14) else "#f0a500" if (s and s >= 8) else "#e05c5c"
        s_str  = f"{s:.0f}/20" if s is not None else "—"
        h += '<div class="comp-row">'
        h += f'<div class="comp-lbl">{c["label"]}<div class="comp-detail">{c["detail"]}</div></div>'
        h += f'<div class="comp-bar-bg"><div class="comp-bar" style="width:{bar_w};background:{bar_c}"></div></div>'
        h += f'<div class="comp-score" style="color:{bar_c}">{s_str}</div>'
        h += '</div>'
    h += '</div></div></div>'

    # ── Key metrics stat cards ─────────────────────────────────────────────────
    spy  = cross.get("SPY", {})
    qqq  = cross.get("QQQ", {})
    tlt  = cross.get("TLT", {})
    gld  = cross.get("GLD", {})
    uup  = cross.get("UUP", {})
    hyg  = cross.get("HYG", {})

    stats = [
        ("SPY 1M",  spy.get("1M"),  None,              "S&P 500"),
        ("QQQ 1M",  qqq.get("1M"),  None,              "Nasdaq 100"),
        ("TLT 1M",  tlt.get("1M"),  None,              "20Y Treasuries"),
        ("Gold 1M", gld.get("1M"),  None,              "GLD ETF"),
        ("DXY 1M",  uup.get("1M"),  None,              "UUP ($ proxy)"),
        ("HYG 1M",  hyg.get("1M"),  None,              "High Yield Bonds"),
    ]

    h += '<div class="stat-grid">'
    for lbl, val, sub, note in stats:
        c = clr(val)
        v_str = fmt_pct(val) if val is not None else "—"
        h += (f'<div class="stat-card">'
              f'<div class="stat-val" style="color:{c}">{v_str}</div>'
              f'<div class="stat-lbl">{lbl}</div>'
              f'<div class="stat-sub">{note}</div>'
              f'</div>')
    h += '</div>'

    # ── Cross-asset table ──────────────────────────────────────────────────────
    h += f'<div class="card-title" style="margin-top:8px">Cross-Asset Performance {help_btn("cross-asset-perf")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>Asset</th><th>Name</th><th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M</th><th class="num">3M</th>'
          '<th class="num">6M</th><th class="num">YTD</th><th class="num">vs 200-day MA</th></tr></thead><tbody>')
    for ticker, name, asset_class in CROSS_ASSET:
        d = cross.get(ticker, {})
        price = d.get("price")
        price_str = (f"${price:,.0f}" if price >= 1000 else f"${price:.2f}") if price else "—"
        def _td(v):
            c = clr(v)
            return f'<td class="num" style="color:{c};font-weight:600">{fmt_pct(v)}</td>'
        vs200     = d.get("vs_200")
        vs200_c   = clr(vs200)
        vs200_str = fmt_pct(vs200)
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td class="num">{price_str}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_td(d.get("1W"))}{_td(d.get("1M"))}{_td(d.get("3M"))}'
              f'{_td(d.get("6M"))}{_td(d.get("YTD"))}'
              f'<td class="num" style="color:{vs200_c};font-weight:600">{vs200_str}</td></tr>')
    h += '</tbody></table></div>'
    return h


def _tab_rates(treasury: dict, yf_yields: dict) -> str:
    h = ""
    cur = treasury.get("current", {})
    hist = treasury.get("history", [])

    # ── Key rates stat cards ───────────────────────────────────────────────────
    maturities = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    h += '<div class="stat-grid">'
    for m in maturities:
        v = cur.get(m)
        h += (f'<div class="stat-card">'
              f'<div class="stat-val" style="color:var(--accent)">{f"{v:.2f}%" if v else "—"}</div>'
              f'<div class="stat-lbl">{m} Treasury</div></div>')
    # 10Y-2Y spread
    y10, y2 = cur.get("10Y"), cur.get("2Y")
    spread = round(y10 - y2, 2) if (y10 and y2) else None
    spread_c = "#00c896" if (spread and spread > 0) else "#e05c5c"
    h += (f'<div class="stat-card">'
          f'<div class="stat-val" style="color:{spread_c}">{fmt_pct(spread, plus=True) if spread is not None else "—"}</div>'
          f'<div class="stat-lbl">10Y – 2Y Spread</div></div>')
    h += '</div>'

    # ── Current yield curve chart ──────────────────────────────────────────────
    h += '<div class="two-col">'
    h += f'<div class="chart-card"><h3>Yield Curve — Spot (Today) {help_btn("yield-curve-spot")}</h3>'
    h += '<div class="chart-wrap" style="height:240px"><canvas id="chart-curve"></canvas></div></div>'

    # ── 10Y-2Y spread history ──────────────────────────────────────────────────
    h += f'<div class="chart-card"><h3>10Y – 2Y Spread History (12 Months) {help_btn("spread-history")}</h3>'
    h += '<div class="chart-wrap" style="height:240px"><canvas id="chart-spread"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Negative (inverted) = recession watch. Source: US Treasury.</p>'
    h += '</div></div>'

    # ── Historical 10Y vs 2Y vs 3M yields ─────────────────────────────────────
    h += f'<div class="chart-card" style="margin-top:20px"><h3>Treasury Yields — 12-Month History {help_btn("treasury-history")}</h3>'
    h += '<div class="chart-wrap" style="height:260px"><canvas id="chart-yields-hist"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Source: yfinance (^IRX, ^TNX, ^TYX) + US Treasury API.</p>'
    h += '</div>'

    # ── Yield curve table ──────────────────────────────────────────────────────
    h += '<div class="sec-hdr">Full Yield Curve</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Maturity</th><th class="num">Yield (%)</th><th class="num">vs 1 Week Ago</th></tr></thead><tbody>'
    # Get value from 5 trading days ago (index -6 if exists)
    prev_row = hist[-6] if len(hist) >= 6 else {}
    for label, _ in TREASURY_FIELDS:
        val  = cur.get(label)
        prev = prev_row.get(label)
        chg  = round(val - prev, 3) if (val and prev) else None
        chg_c = clr(chg) if chg else "#6b7194"
        h += (f'<tr><td>{label}</td>'
              f'<td class="num" style="color:var(--accent)">{f"{val:.3f}%" if val else "—"}</td>'
              f'<td class="num" style="color:{chg_c}">'
              f'{("+"+str(chg) if chg and chg>0 else str(chg))+"%" if chg is not None else "—"}</td></tr>')
    h += '</tbody></table></div></div>'
    return h


def _tab_volatility(vix: dict, cross: dict) -> str:
    h = ""
    vix_val  = vix.get("current")
    vix_avg  = vix.get("avg_1y")
    vvix     = vix.get("vvix")

    # ── VIX stat cards ────────────────────────────────────────────────────────
    vix_c = "#00c896" if (vix_val and vix_val < 18) else "#f0a500" if (vix_val and vix_val < 25) else "#e05c5c"
    vix_lbl = "Low (Complacency)" if (vix_val and vix_val < 16) else "Normal" if (vix_val and vix_val < 20) else "Elevated" if (vix_val and vix_val < 28) else "High Fear"

    h += '<div class="stat-grid">'
    h += (f'<div class="stat-card"><div class="stat-val" style="color:{vix_c}">'
          f'{vix_val:.1f}</div><div class="stat-lbl">VIX (Current)</div>'
          f'<div class="stat-sub">{vix_lbl}</div></div>')
    h += (f'<div class="stat-card"><div class="stat-val" style="color:var(--muted)">'
          f'{vix_avg:.1f}</div><div class="stat-lbl">VIX 1Y Avg</div>'
          f'<div class="stat-sub">Historical baseline</div></div>')
    if vix_val and vix_avg:
        vs_avg = round(vix_val - vix_avg, 1)
        va_c   = "#e05c5c" if vs_avg > 0 else "#00c896"
        h += (f'<div class="stat-card"><div class="stat-val" style="color:{va_c}">'
              f'{vs_avg:+.1f}</div><div class="stat-lbl">VIX vs 1Y Avg</div>'
              f'<div class="stat-sub">Pts above/below mean</div></div>')
    if vvix:
        vvix_c = "#e05c5c" if vvix > 120 else "#f0a500" if vvix > 100 else "#00c896"
        h += (f'<div class="stat-card"><div class="stat-val" style="color:{vvix_c}">'
              f'{vvix:.0f}</div><div class="stat-lbl">VVIX</div>'
              f'<div class="stat-sub">Vol-of-vol (VIX options)</div></div>')

    # HYG and TLT stat cards
    hyg_1m = cross.get("HYG", {}).get("1M")
    lqd_1m = cross.get("LQD", {}).get("1M")
    tlt_1m = cross.get("TLT", {}).get("1M")
    for lbl, val, note in [
        ("HYG 1M", hyg_1m, "High Yield Bonds"),
        ("LQD 1M", lqd_1m, "IG Bonds"),
        ("TLT 1M", tlt_1m, "20Y Treasuries"),
    ]:
        c = clr(val)
        h += (f'<div class="stat-card"><div class="stat-val" style="color:{c}">'
              f'{fmt_pct(val)}</div><div class="stat-lbl">{lbl}</div>'
              f'<div class="stat-sub">{note}</div></div>')
    h += '</div>'

    # ── Charts ────────────────────────────────────────────────────────────────
    h += '<div class="two-col">'
    h += f'<div class="chart-card"><h3>VIX — 12-Month History {help_btn("vix-history")}</h3>'
    h += '<div class="chart-wrap" style="height:250px"><canvas id="chart-vix"></canvas></div>'
    h += f'<p class="note" style="margin-top:8px">Dashed line = 1Y average ({vix_avg:.1f}). Grey band = normal range 15–25.</p>'
    h += '</div>'

    h += f'<div class="chart-card"><h3>HYG vs LQD — 6-Month Price (Normalized to 100) {help_btn("hyg-lqd")}</h3>'
    h += '<div class="chart-wrap" style="height:250px"><canvas id="chart-credit"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Both normalised to 100 at start of window. Divergence = credit stress.</p>'
    h += '</div></div>'

    # ── VIX interpretation guide ───────────────────────────────────────────────
    h += '<div class="card" style="margin-top:4px">'
    h += f'<div class="card-title">VIX Interpretation Guide {help_btn("vix-guide")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += '<thead><tr><th>VIX Range</th><th>Regime</th><th>Interpretation</th></tr></thead><tbody>'
    rows = [
        ("< 15",    "#00c896", "Complacency",    "Market calm, possibly over-extended. Low hedging demand."),
        ("15 – 20", "#4fc08d", "Normal",         "Typical bull market volatility. Healthy risk appetite."),
        ("20 – 28", "#f0a500", "Elevated",        "Uncertainty rising. Investors beginning to hedge."),
        ("28 – 40", "#e07b39", "High Fear",      "Significant stress. Institutional hedging in full swing."),
        ("> 40",    "#e05c5c", "Extreme Fear",   "Crisis-level vol. Historically a contrarian buy signal."),
    ]
    for rng, c, regime, interp in rows:
        bg = "var(--sf2)" if (vix_val and _vix_in_range(vix_val, rng)) else "transparent"
        h += f'<tr style="background:{bg}"><td style="color:{c};font-weight:700">{rng}</td><td style="color:{c}">{regime}</td><td>{interp}</td></tr>'
    h += '</tbody></table></div></div>'
    return h

def _vix_in_range(v, rng):
    try:
        if rng.startswith("<"):  return v < float(rng[1:])
        if rng.startswith(">"):  return v > float(rng[1:])
        lo, hi = rng.split("–")
        return float(lo) <= v < float(hi)
    except Exception:
        return False


def _tab_sectors(sectors: dict, cap_size: dict) -> str:
    h = ""

    def _hcell(v):
        """Color-coded table cell — background intensity scales with magnitude."""
        if v is None:
            return '<td class="num" style="color:var(--muted)">—</td>'
        c = clr(v)
        alpha = min(0.32, abs(v) / 12 * 0.32)
        bg = f"rgba(0,200,150,{alpha:.3f})" if v >= 0 else f"rgba(224,92,92,{alpha:.3f})"
        return f'<td class="num" style="color:{c};font-weight:600;background:{bg}">{fmt_pct(v)}</td>'

    # ── Consolidated heatmap table (sorted by 1M) ──────────────────────────────
    h += f'<div class="card-title">Sector Performance — All Periods {help_btn("sector-perf")}</div>'
    sorted_sectors = sorted(
        SECTORS.items(),
        key=lambda kv: (sectors.get(kv[0], {}).get("1M") or -999),
        reverse=True,
    )
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>ETF</th><th>Sector</th><th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M ↓</th><th class="num">3M</th><th class="num">YTD</th></tr></thead><tbody>')
    for ticker, name in sorted_sectors:
        d     = sectors.get(ticker, {})
        price = d.get("price", 0)
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td class="num">${price:.2f}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_hcell(d.get("1W"))}{_hcell(d.get("1M"))}'
              f'{_hcell(d.get("3M"))}{_hcell(d.get("YTD"))}</tr>')
    h += '</tbody></table></div>'

    # ── Charts: rotation quadrant + momentum bar ───────────────────────────────
    h += '<div class="sec-hdr" style="margin-top:28px">Rotation &amp; Momentum</div>'
    h += '<div class="two-col">'

    # Rotation quadrant (scatter — JS in _build_charts_js)
    h += '<div class="chart-card">'
    h += f'<h3>Rotation Quadrant — 1M vs 3M Return {help_btn("rotation-quadrant")}</h3>'
    h += ('<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:10px;'
          'color:var(--muted);margin-bottom:10px">'
          '<span style="color:#00c896">▲ Leading (top-right)</span>'
          '<span style="color:#4f8ef7">◀ Improving (top-left)</span>'
          '<span style="color:#f0a500">▼ Weakening (bot-right)</span>'
          '<span style="color:#e05c5c">▶ Lagging (bot-left)</span>'
          '</div>')
    h += '<div class="chart-wrap" style="height:310px"><canvas id="chart-rotation"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">X = 1-Month · Y = 3-Month · Hover for exact values.</p>'
    h += '</div>'

    # Momentum bar (sorted 1M — JS in _build_charts_js)
    h += '<div class="chart-card">'
    h += f'<h3>1-Month Momentum — Ranked Best to Worst {help_btn("momentum-bar")}</h3>'
    h += '<div class="chart-wrap" style="height:310px"><canvas id="chart-sectors"></canvas></div>'
    h += '<p class="note" style="margin-top:8px">Sorted by 1-month return.</p>'
    h += '</div>'

    h += '</div>'  # two-col

    # ── Cap Size & Style Rotation ──────────────────────────────────────────────
    h += '<div class="sec-hdr" style="margin-top:32px">Cap Size &amp; Style Rotation</div>'

    cap_tier_colors = {"large": "#4f8ef7", "mid": "#00c896", "small": "#f0a500", "micro": "#a78bfa"}

    # Size table
    h += f'<div class="card-title">By Market Capitalisation {help_btn("cap-size")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>ETF</th><th>Index</th><th>Tier</th>'
          '<th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M</th><th class="num">3M</th>'
          '<th class="num">6M</th><th class="num">YTD</th></tr></thead><tbody>')
    for ticker, name, tier in CAP_SIZE:
        d     = cap_size.get(ticker, {})
        price = d.get("price", 0)
        tc    = cap_tier_colors.get(tier, "#6b7194")
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td><span class="pill" style="background:{tc}22;color:{tc}">{tier}</span></td>'
              f'<td class="num">${price:.2f}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_hcell(d.get("1W"))}{_hcell(d.get("1M"))}'
              f'{_hcell(d.get("3M"))}{_hcell(d.get("6M"))}{_hcell(d.get("YTD"))}</tr>')
    h += '</tbody></table></div>'

    # Style table
    style_colors = {
        "large-growth": "#4f8ef7", "large-value": "#00c896",
        "mid-growth":   "#60a5fa", "mid-value":   "#4fc08d",
        "small-growth": "#f0a500", "small-value": "#e07b39",
    }
    h += f'<div class="card-title" style="margin-top:20px">Growth vs Value — Across Cap Sizes {help_btn("cap-style")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>ETF</th><th>Style</th><th>Bias</th>'
          '<th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M</th><th class="num">3M</th>'
          '<th class="num">6M</th><th class="num">YTD</th></tr></thead><tbody>')
    for ticker, name, style in CAP_STYLE:
        d          = cap_size.get(ticker, {})
        price      = d.get("price", 0)
        sc         = style_colors.get(style, "#6b7194")
        bias_lbl   = style.split("-")[1].title()
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td><span class="pill" style="background:{sc}22;color:{sc}">{bias_lbl}</span></td>'
              f'<td class="num">${price:.2f}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_hcell(d.get("1W"))}{_hcell(d.get("1M"))}'
              f'{_hcell(d.get("3M"))}{_hcell(d.get("6M"))}{_hcell(d.get("YTD"))}</tr>')
    h += '</tbody></table></div>'

    # Charts
    h += '<div class="two-col" style="margin-top:20px">'
    h += (f'<div class="chart-card">'
          f'<h3>Cap Size — 1Y Normalised (Base = 100) {help_btn("cap-size-chart")}</h3>'
          '<div class="chart-wrap" style="height:260px"><canvas id="chart-cap-size"></canvas></div>'
          '<p class="note" style="margin-top:8px">All ETFs set to 100 at start of window. '
          'Divergence shows whether small or large caps are leading the rally.</p></div>')
    h += (f'<div class="chart-card">'
          f'<h3>Growth / Value Ratio — Large Cap (IVW ÷ IVE) {help_btn("growth-value-ratio")}</h3>'
          '<div class="chart-wrap" style="height:260px"><canvas id="chart-growth-value"></canvas></div>'
          '<p class="note" style="margin-top:8px">Rising = growth outperforming value '
          '(falling rates / tech leadership). Falling = value rotation (rising rates / cyclicals).</p></div>')
    h += '</div>'

    return h


def _tab_macro_signals(buffett: dict, money_mkt: dict, global_idx: dict,
                       cape: dict, recession: dict, credit: dict) -> str:
    h = ""

    # ── Valuation Gauges row ───────────────────────────────────────────────────
    h += '<div class="sec-hdr">Valuation &amp; Cycle Risk — Key Gauges</div>'
    h += '<div class="three-col">'

    # Buffett Index card
    bi_val   = buffett.get("current")
    bi_label = buffett.get("label", "N/A")
    bi_color = buffett.get("color", "#6b7194")
    h += '<div class="card">'
    h += f'<div class="card-title">Buffett Index (Market Cap / GDP) {help_btn("buffett-index")}</div>'
    if bi_val is not None:
        h += (f'<div class="stat-val" style="color:{bi_color};font-size:36px;text-align:center;'
              f'padding:12px 0">{bi_val:.1f}%</div>')
        h += f'<div style="text-align:center;color:{bi_color};font-weight:600;margin-bottom:8px">{bi_label}</div>'
    else:
        h += '<div class="stat-val" style="color:var(--muted);text-align:center">—</div>'
    h += ('<p class="note">Wilshire 5000 / US Nominal GDP. '
          '&lt;100% undervalued · 100-150% overvalued · &gt;150% extreme.</p>')
    h += '</div>'

    # Shiller CAPE card
    cape_val   = cape.get("current")
    cape_avg   = cape.get("avg")
    cape_label = cape.get("label", "N/A")
    cape_color = cape.get("color", "#6b7194")
    h += '<div class="card">'
    h += f'<div class="card-title">Shiller CAPE Ratio (P/E10) {help_btn("shiller-cape")}</div>'
    if cape_val is not None:
        h += (f'<div class="stat-val" style="color:{cape_color};font-size:36px;text-align:center;'
              f'padding:12px 0">{cape_val:.1f}×</div>')
        h += f'<div style="text-align:center;color:{cape_color};font-weight:600;margin-bottom:8px">{cape_label}</div>'
        if cape_avg:
            h += f'<p class="note">Historical avg: {cape_avg:.1f}× &nbsp;·&nbsp; '
            h += f'Current premium: {round((cape_val/cape_avg - 1)*100, 0):.0f}% above mean.</p>'
    else:
        h += '<div class="stat-val" style="color:var(--muted);text-align:center">—</div>'
    h += ('<p class="note">S&amp;P 500 price / 10-yr avg real earnings. '
          'Long-run avg ~17×. &gt;30× = expensive; &gt;40× = extreme.</p>')
    h += '</div>'

    # NY Fed Recession Probability card
    rec_val   = recession.get("current")
    rec_label = recession.get("label", "N/A")
    rec_color = recession.get("color", "#6b7194")
    h += '<div class="card">'
    h += f'<div class="card-title">NY Fed Recession Probability (12M) {help_btn("recession-prob")}</div>'
    if rec_val is not None:
        h += (f'<div class="stat-val" style="color:{rec_color};font-size:36px;text-align:center;'
              f'padding:12px 0">{rec_val:.1f}%</div>')
        h += f'<div style="text-align:center;color:{rec_color};font-weight:600;margin-bottom:8px">{rec_label}</div>'
    else:
        h += '<div class="stat-val" style="color:var(--muted);text-align:center">—</div>'
    h += ('<p class="note">NY Fed probit model (10Y−3M spread). '
          '&lt;10% low · 20-30% elevated · &gt;30% high risk.</p>')
    h += '</div>'

    h += '</div>'  # three-col

    # ── Credit Spreads row ─────────────────────────────────────────────────────
    h += '<div class="sec-hdr">Credit Spreads — Risk Appetite Signal</div>'
    hy_cur = credit.get("hy", {}).get("current")
    ig_cur = credit.get("ig", {}).get("current")
    h += '<div class="two-col">'

    # HY OAS stat + note
    h += '<div class="card">'
    h += f'<div class="card-title">High Yield OAS (BAMLH0A0HYM2) {help_btn("hy-oas")}</div>'
    h += '<div class="stat-grid">'
    if hy_cur is not None:
        hy_c = "#00c896" if hy_cur < 4 else "#f0a500" if hy_cur < 7 else "#e05c5c"
        hy_l = "Risk-On (Tight)" if hy_cur < 4 else "Caution" if hy_cur < 7 else "Stress / Risk-Off"
        h += (f'<div class="stat-card"><div class="stat-val" style="color:{hy_c}">'
              f'{hy_cur:.2f}%</div><div class="stat-lbl">HY OAS</div>'
              f'<div class="stat-sub">{hy_l}</div></div>')
    if ig_cur is not None:
        ig_c = "#00c896" if ig_cur < 1.5 else "#f0a500" if ig_cur < 2.5 else "#e05c5c"
        ig_l = "Risk-On" if ig_cur < 1.5 else "Caution" if ig_cur < 2.5 else "Stress"
        h += (f'<div class="stat-card"><div class="stat-val" style="color:{ig_c}">'
              f'{ig_cur:.2f}%</div><div class="stat-lbl">IG OAS</div>'
              f'<div class="stat-sub">{ig_l}</div></div>')
    h += '</div>'
    h += ('<p class="note">ICE BofA Option-Adjusted Spreads vs Treasuries. HY: '
          '&lt;4% risk-on · 4-7% caution · &gt;7% stress. '
          'IG: &lt;1.5% tight · &gt;2.5% widening = risk-off.</p>')
    h += '</div>'

    h += (f'<div class="chart-card">'
          f'<h3>HY &amp; IG Credit Spreads — 5-Year History {help_btn("credit-spread-chart")}</h3>'
          '<div class="chart-wrap" style="height:200px"><canvas id="chart-oas"></canvas></div>'
          '<p class="note" style="margin-top:8px">Spread spikes = credit stress / risk-off. '
          'Proxy via HYG/IGSB trailing yield vs Treasury benchmark (FRED unavailable).</p>'
          '</div>')
    h += '</div>'  # two-col

    # ── Valuation charts two-col ───────────────────────────────────────────────
    h += '<div class="two-col" style="margin-top:4px">'
    h += (f'<div class="chart-card"><h3>Buffett Index — 10-Year History {help_btn("buffett-chart")}</h3>'
          '<div class="chart-wrap" style="height:220px"><canvas id="chart-buffett"></canvas></div>'
          '<p class="note" style="margin-top:8px">Source: Wilshire 5000 (^W5000) / US Nominal GDP (FRED). '
          'Values ~15–20 pts above traditional full-cap calculation — use for trend context.</p>'
          '</div>')
    h += (f'<div class="chart-card"><h3>Shiller CAPE — History {help_btn("cape-chart")}</h3>'
          '<div class="chart-wrap" style="height:220px"><canvas id="chart-cape"></canvas></div>'
          '<p class="note" style="margin-top:8px">S&amp;P 500 CAPE ratio. '
          'Dashed line = long-run average. Source: FRED / Robert Shiller.</p>'
          '</div>')
    h += '</div>'

    # Recession prob chart (full width)
    h += (f'<div class="chart-card" style="margin-bottom:20px">'
          f'<h3>NY Fed 12-Month Recession Probability {help_btn("recession-chart")}</h3>'
          '<div class="chart-wrap" style="height:180px"><canvas id="chart-recession"></canvas></div>'
          '<p class="note" style="margin-top:8px">Shaded area = NBER recession periods. '
          'Source: NY Fed probit model (10Y−3M spread via yfinance ^TNX/^IRX). ~3-month lag.</p>'
          '</div>')

    # ── Cash on Sidelines ──────────────────────────────────────────────────────
    h += '<div class="sec-hdr">Cash on Sidelines — Money Market Fund Assets</div>'
    mm_val = money_mkt.get("current")
    mm_chg = money_mkt.get("change_yoy")

    h += '<div class="two-col">'
    h += f'<div class="card"><div class="card-title">US Money Market Fund AUM {help_btn("money-markets")}</div>'
    h += '<div class="stat-grid">'
    if mm_val is not None:
        mm_c = "#00c896" if (mm_chg and mm_chg > 0) else "#e05c5c"
        h += (f'<div class="stat-card"><div class="stat-val" style="color:var(--accent)">'
              f'${mm_val:.2f}T</div><div class="stat-lbl">Total AUM</div>'
              f'<div class="stat-sub">All money market funds</div></div>')
        if mm_chg is not None:
            h += (f'<div class="stat-card"><div class="stat-val" style="color:{mm_c}">'
                  f'{("+" if mm_chg >= 0 else "")}{mm_chg:.2f}T</div>'
                  f'<div class="stat-lbl">YoY Change ($T)</div>'
                  f'<div class="stat-sub">Cash inflows / outflows</div></div>')
    else:
        h += ('<div class="stat-card"><div class="stat-val" style="color:var(--muted)">—</div>'
              '<div class="stat-lbl">AUM</div><div class="stat-sub">Data unavailable</div></div>')
    h += '</div>'
    h += ('<p class="note" style="margin-top:8px">Rising balances = growing risk aversion / '
          '"dry powder". Record highs often precede equity rallies as cash rotates '
          'back into risk assets.</p>')
    h += '</div>'

    h += (f'<div class="chart-card">'
          f'<h3>Money Market Fund Assets — History ($T) {help_btn("money-markets")}</h3>'
          '<div class="chart-wrap" style="height:200px"><canvas id="chart-mktmoney"></canvas></div>'
          '<p class="note" style="margin-top:8px">Source: Federal Reserve Z.1 L.121 '
          '(FL634090005.Q, quarterly). Lags ~1 quarter.</p>'
          '</div>')
    h += '</div>'  # two-col

    # ── Global Market Indices ───────────────────────────────────────────────────
    h += '<div class="sec-hdr">Global Market Indices — Comprehensive Comparison</div>'

    def _td(v):
        c = clr(v)
        return f'<td class="num" style="color:{c};font-weight:600">{fmt_pct(v)}</td>'

    region_colors = {
        "us":        "#4f8ef7",
        "developed": "#00c896",
        "emerging":  "#f0a500",
        "global":    "#a78bfa",
    }

    # Comprehensive heatmap table — sorted by 1M
    global_sorted = sorted(
        GLOBAL_INDICES,
        key=lambda x: global_idx.get(x[0], {}).get("1M") or -999,
        reverse=True,
    )
    h += f'<div class="card"><div class="card-title">All Major Indices — 1W / 1M / 3M / 6M / YTD {help_btn("global-indices")}</div>'
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>ETF</th><th>Market</th><th>Region</th>'
          '<th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M ↓</th><th class="num">3M</th>'
          '<th class="num">6M</th><th class="num">YTD</th></tr></thead><tbody>')
    for ticker, name, region in global_sorted:
        d         = global_idx.get(ticker, {})
        price     = d.get("price")
        price_str = f"${price:.2f}" if price else "—"
        rc        = region_colors.get(region, "#6b7194")
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td><span class="pill" style="background:{rc}22;color:{rc}">{region}</span></td>'
              f'<td class="num">{price_str}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_td(d.get("1W"))}{_td(d.get("1M"))}{_td(d.get("3M"))}'
              f'{_td(d.get("6M"))}{_td(d.get("YTD"))}</tr>')
    h += '</tbody></table></div></div>'

    # Charts
    h += '<div class="two-col" style="margin-top:16px">'
    h += (f'<div class="chart-card">'
          f'<h3>QQQ vs Global Indices — 1Y Normalised (Base = 100) {help_btn("global-norm-chart")}</h3>'
          '<div class="chart-wrap" style="height:360px"><canvas id="chart-global-norm"></canvas></div>'
          '<p class="note" style="margin-top:8px">All set to 100 at start of window. '
          'QQQ (bold blue) is the growth benchmark. Dashed lines = broad aggregates (EFA, EEM). '
          'Sampled weekly for clarity.</p></div>')
    h += (f'<div class="chart-card">'
          f'<h3>Global 1M Returns — Ranked Best to Worst {help_btn("global-bar-chart")}</h3>'
          '<div class="chart-wrap" style="height:300px"><canvas id="chart-global-bar"></canvas></div>'
          '<p class="note" style="margin-top:8px">1-month return across all tracked '
          'global indices, colour-coded by region.</p></div>')
    h += '</div>'

    h += (f'<div class="chart-card" style="margin-top:16px">'
          f'<h3>SPY / EFA Ratio — US vs Developed World {help_btn("us-vs-intl")}</h3>'
          '<div class="chart-wrap" style="height:200px"><canvas id="chart-spy-efa-ratio"></canvas></div>'
          '<p class="note" style="margin-top:8px">Rising = US outperforming. '
          'Falling = rotation to international. Key driver: USD direction and earnings divergence.</p></div>')

    return h


def _tab_commodities(commodities: dict) -> str:
    h = ""

    # ── Performance table ──────────────────────────────────────────────────────
    h += f'<div class="sec-hdr">Commodity Performance {help_btn("commodity-perf")}</div>'

    def _hcell(v):
        if v is None:
            return '<td class="num" style="color:var(--muted)">—</td>'
        c = clr(v)
        alpha = min(0.28, abs(v) / 15 * 0.28)
        bg = f"rgba(0,200,150,{alpha:.3f})" if v >= 0 else f"rgba(224,92,92,{alpha:.3f})"
        return f'<td class="num" style="color:{c};font-weight:600;background:{bg}">{fmt_pct(v)}</td>'

    cat_colors = {
        "precious":    "#f0a500",
        "energy":      "#4f8ef7",
        "industrial":  "#00c896",
        "agriculture": "#a78bfa",
    }
    h += '<div class="tbl-wrap"><table>'
    h += ('<thead><tr><th>ETF</th><th>Commodity</th><th>Category</th>'
          '<th class="num">Price</th>'
          '<th class="num" style="color:#00c896">1D</th>'
          '<th class="num">1W</th><th class="num">1M</th>'
          '<th class="num">3M</th><th class="num">6M</th>'
          '<th class="num">1Y</th><th class="num">YTD</th></tr></thead><tbody>')
    for ticker, name, cat in COMMODITIES:
        d     = commodities.get(ticker, {})
        price = d.get("price")
        c_col = cat_colors.get(cat, "#6b7194")
        h += (f'<tr><td class="sym">{ticker}</td><td>{name}</td>'
              f'<td><span class="pill" style="background:{c_col}22;color:{c_col}">{cat}</span></td>'
              f'<td class="num">{"$"+f"{price:.2f}" if price else "—"}</td>'
              f'{_live1d_cell(ticker, d.get("1D"))}'
              f'{_hcell(d.get("1W"))}{_hcell(d.get("1M"))}{_hcell(d.get("3M"))}'
              f'{_hcell(d.get("6M"))}{_hcell(d.get("1Y"))}{_hcell(d.get("YTD"))}</tr>')
    h += '</tbody></table></div>'

    # ── Charts ─────────────────────────────────────────────────────────────────
    h += '<div class="sec-hdr" style="margin-top:24px">Commodity Charts</div>'
    h += '<div class="two-col">'

    # Normalized 1Y performance chart
    h += (f'<div class="chart-card">'
          f'<h3>All Commodities — 1Y Normalized (Base = 100) {help_btn("commodity-normalized")}</h3>'
          '<div class="chart-wrap" style="height:280px"><canvas id="chart-comm-norm"></canvas></div>'
          '<p class="note" style="margin-top:8px">All ETFs set to 100 at start of 12-month window. '
          'Sampled weekly.</p>'
          '</div>')

    # Gold/Copper ratio
    h += (f'<div class="chart-card">'
          f'<h3>Gold / Copper Ratio — Risk Sentiment {help_btn("gold-copper")}</h3>'
          '<div class="chart-wrap" style="height:280px"><canvas id="chart-gold-copper"></canvas></div>'
          '<p class="note" style="margin-top:8px">'
          'Rising = risk-off (gold outperforms "Dr. Copper"). '
          'Falling = risk-on / economic optimism.</p>'
          '</div>')
    h += '</div>'

    # Gold/Silver ratio (full width)
    h += ('<div class="two-col" style="margin-top:20px">'
          f'<div class="chart-card">'
          f'<h3>Gold / Silver Ratio {help_btn("gold-silver")}</h3>'
          '<div class="chart-wrap" style="height:220px"><canvas id="chart-gold-silver"></canvas></div>'
          '<p class="note" style="margin-top:8px">'
          '&gt;80× = silver underperforming (risk aversion / deflation). '
          '&lt;60× = silver outperforming (industrial demand / inflation).</p>'
          '</div>'
          f'<div class="chart-card">'
          f'<h3>Oil vs Natural Gas — Relative Energy {help_btn("oil-gas")}</h3>'
          '<div class="chart-wrap" style="height:220px"><canvas id="chart-oil-gas"></canvas></div>'
          '<p class="note" style="margin-top:8px">'
          'Both normalized to 100. Divergence reflects supply shocks and seasonal demand.</p>'
          '</div>'
          '</div>')

    # Interpretation guide
    h += f'<div class="sec-hdr" style="margin-top:24px">Commodity Macro Signals {help_btn("commodity-signals")}</div>'
    h += '<div class="card"><div class="tbl-wrap"><table>'
    h += '<thead><tr><th>Commodity</th><th>Macro Signal</th><th>Rising = </th><th>Falling = </th></tr></thead><tbody>'
    signals = [
        ("Gold",         "#f0a500", "Inflation hedge / Safe haven",       "Risk-off, inflation fears, USD weakness",  "Risk-on, deflation, strong USD"),
        ("Silver",       "#9ca3af", "Industrial + precious hybrid",        "Inflation + economic growth",              "Demand contraction or risk aversion"),
        ("Copper",       "#00c896", "Economic activity (Dr. Copper)",      "GDP growth, construction, manufacturing",  "Economic slowdown or recession"),
        ("WTI Oil",      "#4f8ef7", "Economic activity + energy inflation","Demand recovery, supply tightness",        "Demand weakness or supply glut"),
        ("Natural Gas",  "#60a5fa", "Energy / seasonal / LNG exports",    "Cold weather, LNG demand, supply cuts",    "Mild weather, oversupply, demand decline"),
        ("Agriculture",  "#a78bfa", "Food inflation / supply chain",       "Drought, conflict, supply disruption",     "Good harvests, easing supply"),
    ]
    for name, c, signal, rising, falling in signals:
        h += (f'<tr><td style="color:{c};font-weight:700">{name}</td><td>{signal}</td>'
              f'<td style="color:var(--up)">{rising}</td>'
              f'<td style="color:var(--down)">{falling}</td></tr>')
    h += '</tbody></table></div></div>'

    return h


def _build_charts_js(regime, cross, vix, treasury, yf_yields, sectors, buffett, money_mkt, global_idx,
                     cape, recession, credit, commodities, cap_size) -> str:
    js = ""

    # ── 1. Regime gauge ────────────────────────────────────────────────────────
    score    = regime["score"]
    gauge_c  = regime["color"]
    js += f"""
var gCtx = document.getElementById('gauge-chart');
if(gCtx){{
  new Chart(gCtx,{{
    type:'doughnut',
    data:{{datasets:[{{
      data:[{score},{100-score}],
      backgroundColor:['{gauge_c}','#1c2030'],
      borderWidth:0,
      circumference:180,
      rotation:-90
    }}]}},
    options:{{plugins:{{legend:{{display:false}},tooltip:{{enabled:false}}}},cutout:'75%'}}
  }});
}}
"""

    # ── Charts 2-4: Rates & Yield Curve tab (deferred until tab is opened) ────
    js += "\n_defer('rates',function(){\n"

    # ── 2. Spot yield curve ────────────────────────────────────────────────────
    cur = treasury.get("current", {})
    curve_labels = [l for l, _ in TREASURY_FIELDS if cur.get(l) is not None]
    curve_vals   = [cur.get(l) for l in curve_labels]
    curve_colors = ["#e05c5c" if v < (cur.get("3M") or 5) else "#4f8ef7" for v in curve_vals]
    js += f"""
var cCtx = document.getElementById('chart-curve');
if(cCtx){{
  new Chart(cCtx,{{
    type:'bar',
    data:{{
      labels:{json.dumps(curve_labels)},
      datasets:[{{
        label:'Yield (%)',
        data:{json.dumps(curve_vals)},
        backgroundColor:'#4f8ef7aa',
        borderColor:'#4f8ef7',
        borderWidth:1,
        borderRadius:4
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:function(c){{return c.parsed.y.toFixed(3)+'%';}}}}}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(2)+'%'}}}}}}
      }}
    }}
  }});
}}
"""

    # ── 3. 10Y-2Y spread history ───────────────────────────────────────────────
    hist = treasury.get("history", [])
    # Sample weekly
    sampled = hist[::5] if len(hist) > 60 else hist
    spread_dates  = [r["date"] for r in sampled if r.get("10Y") and r.get("2Y")]
    spread_values = [round(r["10Y"] - r["2Y"], 3) for r in sampled if r.get("10Y") and r.get("2Y")]
    spread_colors = ["rgba(0,200,150,0.15)" if v >= 0 else "rgba(224,92,92,0.15)" for v in spread_values]
    js += f"""
var spCtx = document.getElementById('chart-spread');
if(spCtx){{
  new Chart(spCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(spread_dates)},
      datasets:[{{
        label:'10Y-2Y Spread (%)',
        data:{json.dumps(spread_values)},
        borderColor:'#4f8ef7',
        borderWidth:2,
        pointRadius:0,
        fill:{{target:'origin',above:'rgba(0,200,150,0.12)',below:'rgba(224,92,92,0.18)'}},
        tension:0.3
      }},{{
        label:'Zero line',
        data:new Array({len(spread_dates)}).fill(0),
        borderColor:'#6b7194',
        borderWidth:1,
        borderDash:[4,4],
        pointRadius:0
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(2)+'%'}}}}}}
      }}
    }}
  }});
}}
"""

    # ── 4. Historical yields (10Y, 2Y, 3M from Treasury + yfinance) ───────────
    # Use Treasury API data for 10Y and 2Y (most accurate), yfinance for 3M
    y10_data = [(r["date"], r.get("10Y")) for r in sampled if r.get("10Y")]
    y2_data  = [(r["date"], r.get("2Y"))  for r in sampled if r.get("2Y")]
    y3m_yf   = yf_yields.get("3M", {})
    # Sample yfinance 3M weekly too
    yf_3m_d  = y3m_yf.get("dates", [])[::5]
    yf_3m_v  = y3m_yf.get("values", [])[::5]
    js += f"""
var yhCtx = document.getElementById('chart-yields-hist');
if(yhCtx){{
  new Chart(yhCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps([x[0] for x in y10_data])},
      datasets:[
        {{label:'10Y',data:{json.dumps([x[1] for x in y10_data])},borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:0.2}},
        {{label:'2Y', data:{json.dumps([x[1] for x in y2_data])}, borderColor:'#f0a500',borderWidth:2,pointRadius:0,tension:0.2}},
        {{label:'3M (yf)',data:{json.dumps(yf_3m_v)},borderColor:'#00c896',borderWidth:1.5,pointRadius:0,tension:0.2}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:12}}}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(2)+'%'}}}}}}
      }}
    }}
  }});
}}
"""

    # ── Close rates deferred / open volatility deferred ───────────────────────
    js += "\n});\n"
    js += "\n_defer('volatility',function(){\n"

    # ── 5. VIX history ────────────────────────────────────────────────────────
    vix_dates  = vix.get("dates", [])[::3]   # sample every 3rd day
    vix_values = vix.get("values", [])[::3]
    vix_avg    = vix.get("avg_1y", 20)
    js += f"""
var vCtx = document.getElementById('chart-vix');
if(vCtx){{
  new Chart(vCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(vix_dates)},
      datasets:[
        {{label:'VIX',data:{json.dumps(vix_values)},borderColor:'#e05c5c',borderWidth:2,
          pointRadius:0,tension:0.2,fill:{{target:'origin',above:'rgba(224,92,92,0.08)'}}}},
        {{label:'1Y Avg',data:new Array({len(vix_dates)}).fill({vix_avg}),
          borderColor:'#6b7194',borderWidth:1.5,borderDash:[5,5],pointRadius:0}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:12}}}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}}}}
      }}
    }}
  }});
}}
"""

    # ── 6. HYG vs LQD normalized ──────────────────────────────────────────────
    hyg_d = cross.get("HYG", {})
    lqd_d = cross.get("LQD", {})
    hyg_closes = hyg_d.get("closes", [])
    lqd_closes = lqd_d.get("closes", [])
    hyg_dates  = hyg_d.get("dates", [])
    # Normalise to 100 at start of window, use last 126 days (6 months)
    def _norm(vals):
        if not vals:
            return []
        base = vals[0]
        return [round(v / base * 100, 3) for v in vals] if base else []
    hyg_norm = _norm(hyg_closes)
    lqd_norm = _norm(lqd_closes)
    js += f"""
var crCtx = document.getElementById('chart-credit');
if(crCtx){{
  new Chart(crCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(hyg_dates)},
      datasets:[
        {{label:'HYG (High Yield)',data:{json.dumps(hyg_norm)},borderColor:'#e05c5c',borderWidth:2,pointRadius:0,tension:0.2}},
        {{label:'LQD (IG)',        data:{json.dumps(lqd_norm)},borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:0.2}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:12}}}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(1)}}}}}}
      }}
    }}
  }});
}}
"""

    # ── Close volatility deferred / open sectors deferred ─────────────────────
    js += "\n});\n"
    js += "\n_defer('sectors',function(){\n"

    # ── 7a. Rotation quadrant scatter ─────────────────────────────────────────
    palette = ["#4f8ef7","#00c896","#f0a500","#e05c5c","#a78bfa",
               "#fb923c","#34d399","#f472b6","#60a5fa","#fbbf24","#94a3b8"]
    scatter_pts = [
        {"ticker": t, "name": name,
         "x": round(sectors.get(t, {}).get("1M") or 0, 2),
         "y": round(sectors.get(t, {}).get("3M") or 0, 2)}
        for t, name in SECTORS.items()
    ]
    js += f"""
var rotCtx = document.getElementById('chart-rotation');
if(rotCtx){{
  const pts = {json.dumps(scatter_pts)};
  const pal = {json.dumps(palette[:len(scatter_pts)])};
  const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
  const xMin=Math.min(...xs)-4, xMax=Math.max(...xs)+4;
  const yMin=Math.min(...ys)-6, yMax=Math.max(...ys)+6;

  // Inline plugin: draw ticker labels above each point
  const tickerLabels = {{
    id:'tickerLabels',
    afterDraw(ch){{
      const c2=ch.ctx;
      ch.data.datasets.forEach((ds,i)=>{{
        if(!ds._tk) return;
        const m=ch.getDatasetMeta(i);
        if(!m||!m.data[0]) return;
        const pt=m.data[0];
        c2.save();
        c2.font='bold 10px -apple-system,sans-serif';
        c2.fillStyle=ds.borderColor;
        c2.textAlign='center';
        c2.textBaseline='bottom';
        c2.fillText(ds._tk, pt.x, pt.y-7);
        c2.restore();
      }});
    }}
  }};

  // Quadrant background shading
  const quadBg = {{
    id:'quadBg',
    beforeDraw(ch){{
      const {{ctx:c2,chartArea:{{left,right,top,bottom}},scales:{{x,y}}}} = ch;
      const cx=x.getPixelForValue(0), cy=y.getPixelForValue(0);
      c2.save();
      // Leading (TR) — subtle green
      c2.fillStyle='rgba(0,200,150,0.04)'; c2.fillRect(cx,top,right-cx,cy-top);
      // Lagging (BL) — subtle red
      c2.fillStyle='rgba(224,92,92,0.04)'; c2.fillRect(left,cy,cx-left,bottom-cy);
      // Improving (TL) + Weakening (BR) — subtle blue/orange
      c2.fillStyle='rgba(79,142,247,0.03)'; c2.fillRect(left,top,cx-left,cy-top);
      c2.fillStyle='rgba(240,165,0,0.03)';  c2.fillRect(cx,cy,right-cx,bottom-cy);
      c2.restore();
    }}
  }};

  new Chart(rotCtx,{{
    type:'scatter',
    plugins:[tickerLabels, quadBg],
    data:{{
      datasets:[
        // Zero-axis lines
        {{label:'',data:[{{x:xMin,y:0}},{{x:xMax,y:0}}],type:'line',borderColor:'#3a3f5288',borderWidth:1,pointRadius:0,showLine:true,tension:0}},
        {{label:'',data:[{{x:0,y:yMin}},{{x:0,y:yMax}}],type:'line',borderColor:'#3a3f5288',borderWidth:1,pointRadius:0,showLine:true,tension:0}},
        // One dataset per sector so each gets its own color + label
        ...pts.map((p,i)=>({{
          label:`${{p.ticker}}: 1M ${{p.x>=0?'+':''}}${{p.x.toFixed(1)}}% / 3M ${{p.y>=0?'+':''}}${{p.y.toFixed(1)}}%`,
          _tk:p.ticker,
          data:[{{x:p.x,y:p.y}}],
          backgroundColor:pal[i]+'99',
          borderColor:pal[i],
          pointRadius:9,
          pointHoverRadius:12,
        }}))
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{
          filter:ctx=>!!ctx.dataset._tk,
          callbacks:{{label:ctx=>ctx.dataset.label}}
        }}
      }},
      scales:{{
        x:{{
          title:{{display:true,text:'1-Month Return (%)',color:'#6b7194',font:{{size:11}}}},
          min:xMin,max:xMax,
          grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)+'%'}}
        }},
        y:{{
          title:{{display:true,text:'3-Month Return (%)',color:'#6b7194',font:{{size:11}}}},
          min:yMin,max:yMax,
          grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)+'%'}}
        }}
      }}
    }}
  }});
}}
"""

    # ── 7b. 1M momentum bar — sorted best to worst ────────────────────────────
    sec_sorted = sorted(SECTORS.items(),
                        key=lambda kv: (sectors.get(kv[0], {}).get("1M") or -999),
                        reverse=True)
    bar_labels = [t for t, _ in sec_sorted]
    bar_1m     = [round(sectors.get(t, {}).get("1M") or 0, 2) for t, _ in sec_sorted]
    bar_colors = ["#00c896cc" if v >= 0 else "#e05c5ccc" for v in bar_1m]
    js += f"""
var sCtx = document.getElementById('chart-sectors');
if(sCtx){{
  new Chart(sCtx,{{
    type:'bar',
    data:{{
      labels:{json.dumps(bar_labels)},
      datasets:[{{
        label:'1M %',
        data:{json.dumps(bar_1m)},
        backgroundColor:{json.dumps(bar_colors)},
        borderRadius:4
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:function(c){{return c.dataset.label+': '+c.parsed.y.toFixed(2)+'%';}}}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(1)+'%'}}}}}}
      }}
    }}
  }});
}}
"""

    # ── 7c. Cap size normalised 1Y ────────────────────────────────────────────
    # Build aligned date axis (use SPY as reference, sample weekly)
    spy_cs    = cap_size.get("SPY", {})
    cs_dates  = spy_cs.get("dates", [])[::5]
    n_cs      = len(cs_dates)

    def _cs_norm(ticker):
        vals = cap_size.get(ticker, {}).get("closes", [])[::5]
        if not vals or vals[0] == 0:
            return [None] * n_cs
        base = vals[0]
        padded = (vals + [None] * n_cs)[:n_cs]
        return [round(v / base * 100, 2) if v else None for v in padded]

    cs_datasets = [
        {"ticker": "SPY",  "label": "SPY (Large)",  "color": "#4f8ef7", "width": 2.5},
        {"ticker": "MDY",  "label": "MDY (Mid)",    "color": "#00c896", "width": 2},
        {"ticker": "IWM",  "label": "IWM (Small)",  "color": "#f0a500", "width": 2},
        {"ticker": "IWC",  "label": "IWC (Micro)",  "color": "#a78bfa", "width": 2},
    ]
    ds_json = json.dumps([
        {"label": d["label"],
         "data":  _cs_norm(d["ticker"]),
         "borderColor": d["color"],
         "borderWidth": d["width"],
         "pointRadius": 0,
         "tension": 0.2,
         "spanGaps": True}
        for d in cs_datasets
    ])
    js += f"""
var csCtx = document.getElementById('chart-cap-size');
if(csCtx){{
  new Chart(csCtx,{{
    type:'line',
    data:{{labels:{json.dumps(cs_dates)},datasets:{ds_json}}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:'#6b7194',boxWidth:12}}}},
        tooltip:{{mode:'index',intersect:false,
          callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y?.toFixed(1)}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)}}}}
      }}
    }}
  }});
}}
"""

    # ── 7d. Growth / Value ratio (IVW ÷ IVE) ─────────────────────────────────
    ivw_d = cap_size.get("IVW", {})
    ive_d = cap_size.get("IVE", {})
    ivw_closes = ivw_d.get("closes", [])[::5]
    ive_closes = ive_d.get("closes", [])[::5]
    gv_dates   = ivw_d.get("dates",  [])[::5]
    gv_len     = min(len(ivw_closes), len(ive_closes))
    gv_ratio   = [
        round(ivw_closes[i] / ive_closes[i], 4) if ive_closes[i] else None
        for i in range(gv_len)
    ]
    js += f"""
var gvCtx = document.getElementById('chart-growth-value');
if(gvCtx){{
  new Chart(gvCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(gv_dates[:gv_len])},
      datasets:[{{
        label:'IVW / IVE (Growth ÷ Value)',
        data:{json.dumps(gv_ratio)},
        borderColor:'#4f8ef7',borderWidth:2.5,
        pointRadius:0,tension:0.2,spanGaps:true,
        fill:{{target:'origin',above:'rgba(79,142,247,0.07)'}}
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>'Growth/Value: '+ctx.parsed.y?.toFixed(3)}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(2)}}}}
      }}
    }}
  }});
}}
"""

    # ── Close sectors deferred ─────────────────────────────────────────────────
    js += "\n});\n"

    # ── Charts 9-12: Macro Signals tab (deferred) ─────────────────────────────
    js += "\n_defer('macro',function(){\n"

    # ── 9. Buffett Index history ───────────────────────────────────────────────
    bi_hist = buffett.get("history", [])
    # Sample to at most 130 pts — monthly data → keep every ~1 point
    step = max(1, len(bi_hist) // 130)
    bi_samp  = bi_hist[::step]
    bi_dates = [r["date"] for r in bi_samp]
    bi_vals  = [r["ratio"] for r in bi_samp]
    bi_n     = len(bi_dates)
    bi_cur   = buffett.get("current", 100)
    js += f"""
var biCtx = document.getElementById('chart-buffett');
if(biCtx){{
  new Chart(biCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(bi_dates)},
      datasets:[
        {{label:'Buffett Index (%)',data:{json.dumps(bi_vals)},
          borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:0.3,fill:false}},
        {{label:'80% — Fair Value',data:new Array({bi_n}).fill(80),
          borderColor:'#00c89677',borderWidth:1,borderDash:[6,4],pointRadius:0}},
        {{label:'100%',data:new Array({bi_n}).fill(100),
          borderColor:'#f0a50077',borderWidth:1,borderDash:[6,4],pointRadius:0}},
        {{label:'150% — Sig. Overvalued',data:new Array({bi_n}).fill(150),
          borderColor:'#e05c5c99',borderWidth:1.5,borderDash:[6,4],pointRadius:0}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:'#6b7194',boxWidth:10,font:{{size:10}}}}}},
        tooltip:{{mode:'index',intersect:false,
          callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+'%'}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:10}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)+'%'}}}}
      }}
    }}
  }});
}}
"""

    # ── 10. Money market fund assets ───────────────────────────────────────────
    mm_hist  = money_mkt.get("history", [])
    mm_dates = [r["date"] for r in mm_hist]
    mm_vals  = [r["assets"] for r in mm_hist]
    js += f"""
var mmCtx = document.getElementById('chart-mktmoney');
if(mmCtx){{
  new Chart(mmCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(mm_dates)},
      datasets:[{{
        label:'MMF Assets ($T)',
        data:{json.dumps(mm_vals)},
        borderColor:'#f0a500',borderWidth:2,
        pointRadius:3,pointBackgroundColor:'#f0a500',
        tension:0.3,
        fill:{{target:'origin',above:'rgba(240,165,0,0.08)'}}
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>'$'+ctx.parsed.y.toFixed(2)+'T'}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>'$'+v.toFixed(1)+'T'}}}}
      }}
    }}
  }});
}}
"""

    # ── 11. Global indices normalised (QQQ vs World) ──────────────────────────
    # Chart subset: QQQ, SPY, EWC, VGK, EWJ, EFA, MCHI, INDA, EEM — 9 lines, weekly
    # Colors chosen from across the spectrum for maximum distinctiveness on dark bg.
    # EFA and EEM are aggregates → dashed to visually subordinate them.
    # (ticker, label, color, borderWidth, borderDash)
    gi_spec = [
        ("QQQ",  "QQQ (US Growth)",    "#3b82f6", 2.5, []),      # blue       — bold benchmark
        ("SPY",  "SPY (US)",           "#22d3ee", 1.8, []),      # cyan
        ("EWC",  "Canada (EWC)",       "#f87171", 1.8, []),      # red
        ("VGK",  "Europe (VGK)",       "#4ade80", 1.8, []),      # green
        ("EWJ",  "Japan (EWJ)",        "#c084fc", 1.8, []),      # purple
        ("MCHI", "China (MCHI)",       "#fbbf24", 1.8, []),      # amber
        ("INDA", "India (INDA)",       "#f472b6", 1.8, []),      # pink
        ("EFA",  "Dev. ex-US (EFA)",   "#2dd4bf", 1.5, [5, 4]), # teal  — dashed aggregate
        ("EEM",  "Emerging (EEM)",     "#fb923c", 1.5, [5, 4]), # orange — dashed aggregate
    ]
    ref_gi   = global_idx.get("QQQ", {})
    gi_dates = ref_gi.get("dates", [])[::5]
    n_gi     = len(gi_dates)

    def _gi_norm(ticker):
        vals = global_idx.get(ticker, {}).get("closes", [])[::5]
        if not vals or vals[0] == 0:
            return [None] * n_gi
        base = vals[0]
        padded = (vals + [None] * n_gi)[:n_gi]
        return [round(v / base * 100, 2) if v else None for v in padded]

    gi_ds_json = json.dumps([
        {"label": lbl,
         "data":  _gi_norm(t),
         "borderColor": col,
         "borderWidth": bw,
         "borderDash":  bd,
         "pointRadius": 0,
         "tension": 0.2,
         "spanGaps": True}
        for t, lbl, col, bw, bd in gi_spec
    ])
    js += f"""
var gnCtx = document.getElementById('chart-global-norm');
if(gnCtx){{
  new Chart(gnCtx,{{
    type:'line',
    data:{{labels:{json.dumps(gi_dates)},datasets:{gi_ds_json}}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:'#6b7194',boxWidth:10,font:{{size:10}}}}}},
        tooltip:{{mode:'index',intersect:false,
          callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y?.toFixed(1)}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)}}}}
      }}
    }}
  }});
}}
"""

    # ── 12. Global 1M returns — horizontal bar, sorted best to worst ──────────
    gi_region_colors = {
        "us":        "#4f8ef7",
        "developed": "#00c896",
        "emerging":  "#f0a500",
        "global":    "#a78bfa",
    }
    gi_pairs = sorted(
        [(name, global_idx.get(t, {}).get("1M") or 0, region)
         for t, name, region in GLOBAL_INDICES],
        key=lambda x: x[1], reverse=True
    )
    gi_bar_labels  = [p[0] for p in gi_pairs]
    gi_bar_vals    = [p[1] for p in gi_pairs]
    gi_bar_colors  = [gi_region_colors.get(p[2], "#6b7194") + "bb" for p in gi_pairs]
    js += f"""
var gbCtx = document.getElementById('chart-global-bar');
if(gbCtx){{
  new Chart(gbCtx,{{
    type:'bar',
    data:{{
      labels:{json.dumps(gi_bar_labels)},
      datasets:[{{
        label:'1M %',
        data:{json.dumps(gi_bar_vals)},
        backgroundColor:{json.dumps(gi_bar_colors)},
        borderRadius:4
      }}]
    }},
    options:{{
      indexAxis:'y',
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>ctx.parsed.x.toFixed(2)+'%'}}}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(1)+'%'}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',font:{{size:10}}}}}}
      }}
    }}
  }});
}}
"""

    # ── 13. SPY / EFA ratio ────────────────────────────────────────────────────
    spy_gi  = global_idx.get("SPY", {})
    efa_gi  = global_idx.get("EFA", {})
    spy_closes_gi = spy_gi.get("closes", [])[::5]
    efa_closes_gi = efa_gi.get("closes", [])[::5]
    spy_dates_gi  = spy_gi.get("dates",  [])[::5]
    sr_len    = min(len(spy_closes_gi), len(efa_closes_gi))
    ratio_vals = [
        round(spy_closes_gi[i] / efa_closes_gi[i], 4) if efa_closes_gi[i] else None
        for i in range(sr_len)
    ]
    js += f"""
var srCtx = document.getElementById('chart-spy-efa-ratio');
if(srCtx){{
  new Chart(srCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(spy_dates_gi[:sr_len])},
      datasets:[{{
        label:'SPY / EFA',
        data:{json.dumps(ratio_vals)},
        borderColor:'#a78bfa',borderWidth:2,pointRadius:0,tension:0.2,spanGaps:true,
        fill:{{target:'origin',above:'rgba(167,139,250,0.07)'}}
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>'SPY/EFA: '+ctx.parsed.y?.toFixed(3)}}}}
      }},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(2)}}}}
      }}
    }}
  }});
}}
"""

    js += "\n});\n"  # close macro defer

    # ── Charts 13-16: Macro Signals — CAPE, Recession, OAS ────────────────────
    js += "\n_defer('macro',function(){\n"

    # ── 13. Shiller CAPE history ───────────────────────────────────────────────
    cape_hist = cape.get("history", [])
    step = max(1, len(cape_hist) // 120)
    cape_samp  = cape_hist[::step]
    cape_dates = [r["date"] for r in cape_samp]
    cape_vals  = [r["cape"] for r in cape_samp]
    cape_avg   = cape.get("avg") or 17.0
    cn = len(cape_dates)
    if cn > 0:
        js += f"""
var capeCtx = document.getElementById('chart-cape');
if(capeCtx){{
  new Chart(capeCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(cape_dates)},
      datasets:[
        {{label:'CAPE',data:{json.dumps(cape_vals)},borderColor:'#a78bfa',borderWidth:2,pointRadius:0,tension:0.3,fill:false}},
        {{label:'Avg ({cape_avg:.1f}×)',data:new Array({cn}).fill({cape_avg}),
          borderColor:'#6b719488',borderWidth:1,borderDash:[6,4],pointRadius:0}},
        {{label:'30× (Expensive)',data:new Array({cn}).fill(30),
          borderColor:'#f0a50066',borderWidth:1,borderDash:[4,4],pointRadius:0}},
        {{label:'40× (Extreme)',data:new Array({cn}).fill(40),
          borderColor:'#e05c5c66',borderWidth:1,borderDash:[4,4],pointRadius:0}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:10,font:{{size:10}}}}}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+'×'}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:10}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)+'×'}}}}}}
    }}
  }});
}}
"""
    else:
        js += """
var capeCtx = document.getElementById('chart-cape');
if(capeCtx){
  var p=capeCtx.parentElement;
  p.style.display='flex';p.style.alignItems='center';p.style.justifyContent='center';
  p.innerHTML='<div style="color:#6b7194;font-size:13px;text-align:center;padding:20px">'
    +'Historical CAPE series not available via free FRED endpoint.<br>'
    +'Current value shown above is S&amp;P 500 trailing P/E (TTM approximation).</div>';
}
"""

    # ── 14. NY Fed Recession Probability ──────────────────────────────────────
    rec_hist = recession.get("history", [])
    step_r   = max(1, len(rec_hist) // 120)
    rec_samp  = rec_hist[::step_r]
    rec_dates = [r["date"] for r in rec_samp]
    rec_vals  = [r["prob"] for r in rec_samp]
    rn = len(rec_dates)
    js += f"""
var recCtx = document.getElementById('chart-recession');
if(recCtx){{
  new Chart(recCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(rec_dates)},
      datasets:[
        {{label:'Recession Prob (%)',data:{json.dumps(rec_vals)},
          borderColor:'#e05c5c',borderWidth:2,pointRadius:0,tension:0.3,
          fill:{{target:'origin',above:'rgba(224,92,92,0.12)'}}}},
        {{label:'30% threshold',data:new Array({rn}).fill(30),
          borderColor:'#f0a50077',borderWidth:1.5,borderDash:[5,5],pointRadius:0}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:10}}}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+'%'}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:10}}}},y:{{min:0,max:100,grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v+'%'}}}}}}
    }}
  }});
}}
"""

    # ── 15. HY + IG OAS credit spreads ────────────────────────────────────────
    hy_hist = credit.get("hy", {}).get("history", [])
    ig_hist = credit.get("ig", {}).get("history", [])
    hy_dates = [r["date"] for r in hy_hist]
    hy_vals  = [r["spread"] for r in hy_hist]
    ig_vals  = [r["spread"] for r in ig_hist][:len(hy_dates)]
    js += f"""
var oasCtx = document.getElementById('chart-oas');
if(oasCtx){{
  new Chart(oasCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(hy_dates)},
      datasets:[
        {{label:'HY OAS (%)',data:{json.dumps(hy_vals)},borderColor:'#e05c5c',borderWidth:2,pointRadius:0,tension:0.2,fill:false}},
        {{label:'IG OAS (%)',data:{json.dumps(ig_vals)},borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:0.2,fill:false}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:12}}}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(2)+'%'}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:10}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(1)+'%'}}}}}}
    }}
  }});
}}
"""

    js += "\n});\n"  # close macro (CAPE/recession/OAS) defer

    # ── Charts 17-20: Commodities tab (deferred) ───────────────────────────────
    js += "\n_defer('commodities',function(){\n"

    comm_palette = {
        "GLD": "#f0a500", "SLV": "#9ca3af", "USO": "#4f8ef7",
        "UNG": "#60a5fa", "CPER": "#00c896", "DBA": "#a78bfa",
    }
    # Aligned date axis from GLD (usually longest series)
    ref_d = commodities.get("GLD", {})
    ref_dates  = ref_d.get("dates",  [])[::5]
    n_dates    = len(ref_dates)

    def _norm_comm(closes, n):
        if not closes or closes[0] == 0:
            return [None] * n
        base = closes[0]
        padded = (closes + [None] * n)[:n]
        return [round(v / base * 100, 2) if v else None for v in padded]

    comm_norm_datasets = []
    for ticker, name, _ in COMMODITIES:
        closes = commodities.get(ticker, {}).get("closes", [])[::5]
        normed = _norm_comm(closes, n_dates)
        col    = comm_palette.get(ticker, "#ffffff")
        comm_norm_datasets.append({
            "label": f"{ticker} ({name})",
            "data": normed,
            "borderColor": col,
            "borderWidth": 2,
            "pointRadius": 0,
            "tension": 0.2,
            "spanGaps": True,
        })

    # Current prices for end-of-line labels
    comm_prices = {t: round(commodities.get(t, {}).get("price") or 0, 2)
                   for t, _, _ in COMMODITIES}

    js += f"""
var cnCtx = document.getElementById('chart-comm-norm');
if(cnCtx){{
  var commPrices = {json.dumps(comm_prices)};
  new Chart(cnCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(ref_dates)},
      datasets:{json.dumps(comm_norm_datasets)}
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      layout:{{padding:{{right:58}}}},
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:10,font:{{size:10}}}}}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y?.toFixed(1)}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)}}}}}}
    }},
    plugins:[{{
      id:'commPriceLabels',
      afterDraw(chart){{
        const ctx=chart.ctx;
        chart.data.datasets.forEach((ds,i)=>{{
          const meta=chart.getDatasetMeta(i);
          if(!meta.visible)return;
          let li=-1;
          for(let j=ds.data.length-1;j>=0;j--){{if(ds.data[j]!=null){{li=j;break;}}}}
          if(li<0)return;
          const pt=meta.data[li];
          if(!pt)return;
          const ticker=ds.label.split(' ')[0];
          const price=commPrices[ticker];
          if(!price)return;
          ctx.save();
          ctx.font='bold 10px DM Mono,monospace';
          ctx.fillStyle=ds.borderColor;
          ctx.textAlign='left';
          ctx.textBaseline='middle';
          ctx.fillText('$'+price.toFixed(2),pt.x+6,pt.y);
          ctx.restore();
        }});
      }}
    }}]
  }});
}}
"""

    # ── 18. Gold/Copper ratio ──────────────────────────────────────────────────
    gld_c  = commodities.get("GLD",  {}).get("closes", [])[::5]
    cper_c = commodities.get("CPER", {}).get("closes", [])[::5]
    slv_c  = commodities.get("SLV",  {}).get("closes", [])[::5]
    uso_c  = commodities.get("USO",  {}).get("closes", [])[::5]
    ung_c  = commodities.get("UNG",  {}).get("closes", [])[::5]
    gc_len = min(len(gld_c), len(cper_c))
    gs_len = min(len(gld_c), len(slv_c))
    og_len = min(len(uso_c), len(ung_c))

    gc_ratio = [round(gld_c[i] / cper_c[i], 3) if cper_c[i] else None for i in range(gc_len)]
    gs_ratio = [round(gld_c[i] / slv_c[i], 2)  if slv_c[i]  else None for i in range(gs_len)]

    uso_norm = _norm_comm(uso_c, og_len)
    ung_norm = _norm_comm(ung_c, og_len)

    js += f"""
var gcCtx = document.getElementById('chart-gold-copper');
if(gcCtx){{
  new Chart(gcCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(ref_dates[:gc_len])},
      datasets:[{{label:'Gold/Copper Ratio',data:{json.dumps(gc_ratio)},
        borderColor:'#f0a500',borderWidth:2,pointRadius:0,tension:0.2,spanGaps:true,
        fill:{{target:'origin',above:'rgba(240,165,0,0.06)'}}}}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>'GLD/CPER: '+ctx.parsed.y?.toFixed(2)}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(1)}}}}}}
    }}
  }});
}}

var gsCtx = document.getElementById('chart-gold-silver');
if(gsCtx){{
  new Chart(gsCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(ref_dates[:gs_len])},
      datasets:[{{label:'Gold/Silver Ratio',data:{json.dumps(gs_ratio)},
        borderColor:'#9ca3af',borderWidth:2,pointRadius:0,tension:0.2,spanGaps:true,
        fill:{{target:'origin',above:'rgba(156,163,175,0.06)'}}}}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>'GLD/SLV: '+ctx.parsed.y?.toFixed(1)}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)}}}}}}
    }}
  }});
}}

var ogCtx = document.getElementById('chart-oil-gas');
if(ogCtx){{
  new Chart(ogCtx,{{
    type:'line',
    data:{{
      labels:{json.dumps(ref_dates[:og_len])},
      datasets:[
        {{label:'Oil (USO)',data:{json.dumps(uso_norm)},borderColor:'#4f8ef7',borderWidth:2,pointRadius:0,tension:0.2,spanGaps:true}},
        {{label:'Nat Gas (UNG)',data:{json.dumps(ung_norm)},borderColor:'#60a5fa',borderWidth:2,pointRadius:0,tension:0.2,spanGaps:true}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{color:'#6b7194',boxWidth:10}}}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y?.toFixed(1)}}}}}},
      scales:{{x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',maxTicksLimit:8}}}},y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:v=>v.toFixed(0)}}}}}}
    }}
  }});
}}
"""

    js += "\n});\n"  # close commodities defer

    # ── 8. Cross-asset 1M performance (horizontal bar, overview tab — direct) ──
    # Sort best → worst so chart reads top-to-bottom in descending order
    ca_pairs  = sorted(
        [(name, cross.get(t, {}).get("1M") or 0) for t, name, _ in CROSS_ASSET],
        key=lambda x: x[1],
        reverse=True,
    )
    ca_names  = [p[0] for p in ca_pairs]
    ca_1m     = [p[1] for p in ca_pairs]
    ca_colors = ["#00c896aa" if v >= 0 else "#e05c5caa" for v in ca_1m]
    js += f"""
var caCtx = document.getElementById('chart-cross-asset');
if(caCtx){{
  new Chart(caCtx,{{
    type:'bar',
    data:{{
      labels:{json.dumps(ca_names)},
      datasets:[{{
        label:'1M %',
        data:{json.dumps(ca_1m)},
        backgroundColor:{json.dumps(ca_colors)},
        borderRadius:4
      }}]
    }},
    options:{{
      indexAxis:'y',
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:function(c){{return c.parsed.x.toFixed(2)+'%';}}}}}}}},
      scales:{{
        x:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194',callback:function(v){{return v.toFixed(1)+'%'}}}}}},
        y:{{grid:{{color:'#252a3a'}},ticks:{{color:'#6b7194'}}}}
      }}
    }}
  }});
}}
"""
    return js


def build_html(regime, cross, vix, treasury, yf_yields, sectors, buffett, money_mkt, global_idx,
               cape, recession, credit, commodities, cap_size) -> str:
    now = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    h  = '<!DOCTYPE html><html lang="en"><head>'
    h += '<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    h += '<title>Macro Dashboard</title>'
    h += f'<style>{_CSS}</style>'
    h += '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
    h += '</head><body>'

    # Header
    score = regime["score"]
    color = regime["color"]
    label = regime["label"]
    h += (f'<div class="hdr">'
          f'<div><div class="hdr-title">🌐 Macro Dashboard</div>'
          f'<div class="hdr-sub">Market regime · rates · volatility · sectors · cross-asset</div></div>'
          f'<div class="pill" style="background:{color}22;color:{color};border:1px solid {color}44;font-size:13px;padding:6px 14px">'
          f'{label} &nbsp; <strong>{score}/100</strong></div>'
          f'<div class="hdr-badge">Updated {now}</div>'
          f'</div>')

    # Tabs
    tabs = [
        ("overview",     "Overview"),
        ("rates",        "Rates & Yield Curve"),
        ("volatility",   "Volatility & Credit"),
        ("sectors",      "Sector Rotation"),
        ("macro",        "Macro Signals"),
        ("commodities",  "Commodities"),
    ]
    h += '<div class="tab-bar">'
    for i, (tid, tlbl) in enumerate(tabs):
        active = " active" if i == 0 else ""
        h += f'<button class="tab-btn{active}" onclick="switchTab(\'{tid}\',this)">{tlbl}</button>'
    h += '</div>'

    h += '<div class="container">'
    h += f'<div id="tab-overview" class="tab-panel active">'
    h += _tab_overview(regime, cross)
    # Cross-asset 1M bar chart
    h += '<div class="sec-hdr">1-Month Performance</div>'
    h += f'<div class="chart-card"><h3>Cross-Asset — 1 Month Return {help_btn("cross-asset-chart")}</h3>'
    h += f'<div class="chart-wrap" style="height:{len(CROSS_ASSET)*36}px"><canvas id="chart-cross-asset"></canvas></div></div>'
    h += _interp_overview(regime, cross)
    h += '</div>'

    h += f'<div id="tab-rates" class="tab-panel">'
    h += _tab_rates(treasury, yf_yields)
    h += _interp_rates(treasury, yf_yields)
    h += '</div>'

    h += f'<div id="tab-volatility" class="tab-panel">'
    h += _tab_volatility(vix, cross)
    h += _interp_volatility(vix, cross, credit)
    h += '</div>'

    h += f'<div id="tab-sectors" class="tab-panel">'
    h += _tab_sectors(sectors, cap_size)
    h += _interp_sectors(sectors)
    h += '</div>'

    h += f'<div id="tab-macro" class="tab-panel">'
    h += _tab_macro_signals(buffett, money_mkt, global_idx, cape, recession, credit)
    h += _interp_macro(buffett, cape, recession, credit, money_mkt, global_idx)
    h += '</div>'

    h += f'<div id="tab-commodities" class="tab-panel">'
    h += _tab_commodities(commodities)
    h += _interp_commodities(commodities)
    h += '</div>'

    h += '</div>'  # container

    # JS
    h += '<script>'
    h += ('window._dc={};\n'
          'function _defer(tab,fn){if(!window._dc[tab])window._dc[tab]=[];window._dc[tab].push(fn);}\n'
          'function switchTab(id,btn){'
          'document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));'
          'document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));'
          'document.getElementById("tab-"+id).classList.add("active");'
          'btn.classList.add("active");'
          'if(window._dc[id]){window._dc[id].forEach(f=>f());delete window._dc[id];}}\n')
    h += _build_charts_js(regime, cross, vix, treasury, yf_yields, sectors, buffett, money_mkt, global_idx,
                          cape, recession, credit, commodities, cap_size)
    # Help popover shared JS
    import json as _json
    help_js_obj = _json.dumps({k: v for k, v in _HELP_DATA.items()})
    h += f'var _HELP={help_js_obj};\n'
    h += """
function showHelp(key, evt) {
  evt.stopPropagation();
  var d = _HELP[key];
  if (!d) return;
  var pop = document.getElementById('help-pop');
  document.getElementById('help-pop-title').textContent = d.title;
  document.getElementById('help-pop-body').innerHTML = d.body;
  pop.style.visibility = 'hidden';
  pop.style.display    = 'block';
  var vw = window.innerWidth, vh = window.innerHeight;
  var pw = pop.offsetWidth,   ph = pop.offsetHeight;
  var x  = evt.clientX + 14;
  var y  = evt.clientY + 14;
  if (x + pw > vw - 8)  x = evt.clientX - pw - 14;
  if (x < 8)            x = 8;
  if (y + ph > vh - 8)  y = evt.clientY - ph - 14;
  if (y < 8)            y = 8;
  pop.style.left       = x + 'px';
  pop.style.top        = y + 'px';
  pop.style.visibility = '';
}
function closeHelp() {
  document.getElementById('help-pop').style.display = 'none';
}
document.addEventListener('click', function(e) {
  var pop = document.getElementById('help-pop');
  if (pop && pop.style.display !== 'none' && !pop.contains(e.target)) closeHelp();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeHelp();
});
"""
    h += '</script>'
    # Help popover DOM element (shared singleton)
    h += ('<div id="help-pop" role="tooltip">'
          '<button id="help-pop-close" onclick="closeHelp()" aria-label="Close">×</button>'
          '<div id="help-pop-title"></div>'
          '<div id="help-pop-body"></div>'
          '</div>')
    # ── Live 1D quote refresher ───────────────────────────────────────────────
    h += """
<script>
(function(){
  /* Collect the unique set of tickers that need live 1D data */
  var cells   = document.querySelectorAll('.live1d[data-ticker]');
  var tickers = Array.from(new Set(Array.from(cells).map(function(c){return c.dataset.ticker;}).filter(Boolean)));
  if(!tickers.length) return;

  var PORTS        = [5050, 5051];
  var workingPort  = null;
  var statusEl     = null;

  /* Inject a small status badge into the page once DOM is ready */
  function ensureStatus(){
    if(statusEl) return;
    statusEl = document.createElement('div');
    statusEl.id = 'live1d-status';
    statusEl.style.cssText = [
      'position:fixed','bottom:14px','right:18px','z-index:9999',
      'font-family:DM Mono,monospace','font-size:10px','letter-spacing:1px',
      'background:rgba(20,24,40,0.92)','border:1px solid rgba(0,200,150,0.35)',
      'border-radius:6px','padding:5px 12px','color:#6b7194',
      'pointer-events:none','transition:opacity .4s'
    ].join(';');
    document.body.appendChild(statusEl);
  }

  function setStatus(msg, color){
    ensureStatus();
    statusEl.style.color  = color || '#6b7194';
    statusEl.textContent  = msg;
    statusEl.style.opacity = '1';
  }

  /* Color + background matching the existing heatmap cell style */
  function styleCell(cell, chg){
    if(chg == null){ cell.textContent='—'; cell.style.color=''; cell.style.background=''; cell.style.fontWeight=''; return; }
    var pos    = chg >= 0;
    var alpha  = Math.min(0.28, Math.abs(chg) / 5 * 0.28);
    cell.textContent  = (pos ? '+' : '') + chg.toFixed(2) + '%';
    cell.style.color  = pos ? '#00c896' : '#e05c5c';
    cell.style.background = pos ? 'rgba(0,200,150,'+alpha+')' : 'rgba(224,92,92,'+alpha+')';
    cell.style.fontWeight = '600';
  }

  function applyData(data){
    tickers.forEach(function(t){
      var chg = (data[t] !== undefined) ? data[t] : null;
      document.querySelectorAll('.live1d[data-ticker="'+t+'"]').forEach(function(cell){ styleCell(cell, chg); });
    });
    var now = new Date();
    var hm  = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
    setStatus('● LIVE · ' + hm, '#00c896');
  }

  async function tryFetch(port){
    var url  = 'http://localhost:' + port + '/api/live-quotes?symbols=' + tickers.join(',');
    var ctrl = new AbortController();
    var tid  = setTimeout(function(){ ctrl.abort(); }, 5000);
    try {
      var resp = await fetch(url, {signal: ctrl.signal});
      clearTimeout(tid);
      if(!resp.ok) throw new Error('status ' + resp.status);
      return await resp.json();
    } catch(e) { clearTimeout(tid); throw e; }
  }

  async function refresh(){
    setStatus('↻ updating…', '#6b7194');
    var ports = workingPort ? [workingPort].concat(PORTS.filter(function(p){ return p!==workingPort; })) : PORTS;
    for(var i=0; i<ports.length; i++){
      try {
        var data = await tryFetch(ports[i]);
        workingPort = ports[i];
        applyData(data);
        return;
      } catch(e) { /* try next port */ }
    }
    /* All ports failed — Flask probably not running */
    setStatus('○ offline', '#6b7194');
  }

  /* Initial fetch + 60-second auto-refresh */
  refresh();
  setInterval(refresh, 60000);
})();
</script>
"""
    h += '</body></html>'
    return h


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Macro environment dashboard",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parser.parse_args()

    print("\n  🌐  Macro Dashboard\n")

    steps = [
        ("[1/14] Fetching Treasury yield curve...",      fetch_treasury_yields),
        ("[2/14] Fetching VIX & volatility data...",     fetch_vix),
        ("[3/14] Fetching cross-asset prices...",        fetch_cross_asset),
        ("[4/14] Fetching sector ETFs...",               fetch_sectors),
        ("[5/14] Fetching yfinance yield history...",    fetch_yf_yield_history),
        ("[6/14] Fetching Buffett Index (FRED)...",      fetch_buffett_index),
        ("[7/14] Fetching money market data (FRED)...",  fetch_money_markets),
        ("[8/14] Fetching global indices (comprehensive)...", fetch_global_indices),
        ("[9/14] Fetching Shiller CAPE...",              fetch_shiller_cape),
        ("[10/14] Fetching recession probability...",    fetch_recession_prob),
        ("[11/14] Fetching credit spreads (FRED)...",    fetch_credit_spreads),
        ("[12/14] Fetching commodity prices...",         fetch_commodities),
        ("[13/14] Fetching cap size ETFs...",            fetch_cap_size),
        ("[14/14] Fetching global equity legacy...",     fetch_global_equity),
    ]

    results = []
    for msg, fn in steps:
        print(f"  {msg}")
        results.append(fn())

    (treasury, vix, cross, sectors, yf_yields,
     buffett, money_mkt, global_idx,
     cape, recession, credit, commodities,
     cap_size, _global_eq_legacy) = results

    print("\n  Computing macro regime score...")
    regime = compute_regime(cross, vix, treasury)
    print(f"  Regime: {regime['label']} ({regime['score']}/100)\n")

    print("  Generating report...")
    html = build_html(regime, cross, vix, treasury, yf_yields, sectors, buffett, money_mkt, global_idx,
                      cape, recession, credit, commodities, cap_size)

    date_prefix = datetime.datetime.now().strftime("%Y_%m_%d")
    outfile     = os.path.join(OUTPUT_DIR, f"{date_prefix}_macro.html")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(outfile) / 1024
    print(f"  ✓  Report saved → {outfile}  ({size_kb:.0f} KB)")
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        print(f"  Opening in browser...\n")
        webbrowser.open(f"file://{os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()
