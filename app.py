#!/usr/bin/env python3
"""
ValuationSuite — Flask Web Launcher
=====================================
Browser-based replacement for the Tkinter launcher.

Usage
-----
    python app.py              # starts server + opens browser
    python app.py --port 5050  # custom port
    python app.py --no-browser # skip auto-open
"""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime

from flask import Flask, Response, jsonify, make_response, request

ROOT = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Deep Dive Tickers ─────────────────────────────────────────────────────────
DEEPDIVE_DIR = os.path.join(ROOT, "deepDiveTickers")
DEEPDIVE_CSV = os.path.join(DEEPDIVE_DIR, "deepDiveTickers.csv")
os.makedirs(DEEPDIVE_DIR, exist_ok=True)

# ── Server port (set in __main__, passed to child processes) ──────────────────
_SERVER_PORT: int = 5050

# ── Script catalogue ──────────────────────────────────────────────────────────

SCRIPTS = {
    "Macro Dashboard": {
        "path": os.path.join(ROOT, "7_macroDashboard", "macroDashboard.py"),
        "desc": "Macro regime dashboard — rates, yield curve, volatility, credit, sector rotation & cross-asset performance.",
        "icon": "🌐",
        "params": [],
    },
    "Value Screener": {
        "path": os.path.join(ROOT, "1_valueScreener", "valueScreener.py"),
        "desc": "Screen S&P 500, Nasdaq 100, Russell 2000, or TSX for undervalued stocks.",
        "icon": "🔍",
        "params": [
            dict(id="index", label="Index",      type="option", flag="--index",
                 required=True,  default="SPX",  values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",   label="Top N",      type="entry",  flag="--top",
                 required=False, default="50"),
            dict(id="csv",   label="Export CSV", type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Growth Screener": {
        "path": os.path.join(ROOT, "2_growthScreener", "growthScreener.py"),
        "desc": "Screen indices for high-growth stocks using a 4-pillar quality + momentum model.",
        "icon": "📈",
        "params": [
            dict(id="index",    label="Index",        type="option", flag="--index",
                 required=True,  default="SPX",  values=["SPX", "NDX", "RUT", "TSX", "SPXRUT"]),
            dict(id="top",      label="Top N",        type="entry",  flag="--top",
                 required=False, default="",
                 hint="Stocks to screen (default: SPX=503 NDX=110 RUT=500)"),
            dict(id="backtest", label="Full Backtest", type="check",  flag="--backtest",
                 required=False, default=False),
            dict(id="backtest_days", label="Backtest days", type="entry", flag="--backtest-days",
                 required=False, default=""),
            dict(id="csv",      label="Export CSV",   type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Valuation Master": {
        "path": os.path.join(ROOT, "3_valuationTool", "valuationMaster.py"),
        "desc": "Deep single-stock valuation — 5-tab HTML report with 20+ methods.",
        "icon": "📊",
        "params": [
            dict(id="ticker",   label="Ticker",        type="entry",  positional=True,
                 required=True,  default="NVDA"),
            dict(id="backtest", label="Backtest days", type="entry",  flag="--backtest",
                 required=False, default="500"),
            dict(id="wacc",     label="WACC override", type="entry",  flag="--wacc",
                 required=False, default=""),
        ],
    },
    "Run Model": {
        "path": os.path.join(ROOT, "run_model.py"),
        "desc": "Run any single valuation model on a ticker with optional backtest plot.",
        "icon": "🔬",
        "params": [
            dict(id="ticker", label="Ticker",          type="entry",  positional=True,
                 required=True,  default="NVDA"),
            dict(id="model",  label="Model",           type="option", positional=True,
                 required=True,  default="dcf",
                 values=["dcf", "three-stage", "monte-carlo", "pfcf", "pe", "ev-ebitda",
                         "fcf-yield", "rim", "roic", "ncav", "mean-reversion",
                         "reverse-dcf", "peg", "ev-ntm", "tam", "rule40", "erg",
                         "scurve", "pie", "ddm", "graham", "multifactor", "bayesian",
                         "wacc", "piotroski", "altman", "all"]),
            dict(id="wacc",   label="WACC override",   type="entry",  flag="--wacc",
                 required=False, default=""),
            dict(id="growth", label="Growth override", type="entry",  flag="--growth",
                 required=False, default=""),
            dict(id="plot",   label="Plot backtest",   type="check",  flag="--plot",
                 required=False, default=False),
            dict(id="days",   label="Plot days",       type="entry",  flag="--days",
                 required=False, default="1000"),
        ],
    },
    "Scatter Plots": {
        "path": os.path.join(ROOT, "5_scatterPlots", "scatterPlots.py"),
        "desc": "Interactive valuation scatter plots — EV/S, P/E, FCF, ROIC and more across an index or custom CSV.",
        "icon": "📉",
        "params": [
            dict(id="index",   label="Index",           type="option", flag="--index",
                 required=False, default="SPX",       values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",     label="Max tickers",    type="entry",  flag="--top",
                 required=False, default="150"),
            dict(id="tickers", label="Tickers (comma-separated — overrides index)",
                 type="entry",  flag="--tickers",
                 required=False, default=""),
            dict(id="csv",     label="Custom CSV",     type="file",   flag="--csv",
                 required=False, default="",
                 mutex_with=["index", "tickers"]),
        ],
    },
    "Sentiment Analyzer": {
        "path": os.path.join(ROOT, "6_sentimentAnalyzer", "sentimentAnalyzer.py"),
        "desc": "Multi-source sentiment — news, social, short interest, insider activity, options, analyst ratings, SEC filings & congressional trading.",
        "icon": "📡",
        "params": [
            dict(id="csvfile", label="Portfolio CSV", type="file",  positional=True,
                 required=True,
                 default=os.path.join(ROOT, "4_portfolioAnalyzer", "deepDiveTickers.csv")),
            dict(id="days",    label="Lookback days", type="entry", flag="--days",
                 required=False, default="30"),
        ],
    },
    "Portfolio Analyzer": {
        "path": os.path.join(ROOT, "4_portfolioAnalyzer", "portfolioAnalyzer.py"),
        "desc": "Multi-stock portfolio dashboard — signals, technicals, rebalancing, backtesting.",
        "icon": "💼",
        "params": [
            dict(id="csvfile",  label="Portfolio CSV", type="file",  positional=True,
                 required=True,
                 default=os.path.join(ROOT, "4_portfolioAnalyzer", "currentlyOwnded.csv")),
            dict(id="backtest", label="Backtest days", type="entry", flag="--backtest",
                 required=False, default="500"),
        ],
    },
    "Watchlist Tracker": {
        "path": os.path.join(ROOT, "8_watchlist", "watchlistTracker.py"),
        "desc": "Watchlist & valuation history — tracks fair value snapshots and portfolio signals over time.",
        "icon": "📋",
        "params": [
            dict(id="add",     label="Add ticker",    type="entry", flag="--add",
                 required=False, default=""),
            dict(id="target",  label="Target price",  type="entry", flag="--target",
                 required=False, default=""),
            dict(id="notes",   label="Notes",         type="entry", flag="--notes",
                 required=False, default=""),
            dict(id="remove",  label="Remove ticker", type="entry", flag="--remove",
                 required=False, default=""),
            dict(id="history", label="History for",   type="entry", flag="--history",
                 required=False, default=""),
        ],
    },
    "Deep Dive List": {
        "url":  "/deep-dive",
        "desc": "Edit the deep dive tickers list — add or remove stocks flagged for deeper analysis.",
        "icon": "🔬",
        "params": [],
    },
    "Research Scanner": {
        "path": os.path.join(ROOT, "9_researchScanner", "researchScanner.py"),
        "desc": "Emerging technology research — FDA pipeline, Phase 2/3 clinical trials, arXiv papers, IPO S-1 filings & USPTO patents.",
        "icon": "🔭",
        "params": [
            dict(id="days", label="Lookback (days)", type="entry", flag="--days",
                 required=False, default="60"),
        ],
    },
    "Topic Explorer": {
        "path": os.path.join(ROOT, "9_researchScanner", "topicExplorer.py"),
        "desc": "Interactive topic explorer — select from 25 research themes across AI, biotech, semiconductors, health tech, energy & more. Scans arXiv, NIH grants & clinical trials.",
        "icon": "🧭",
        "params": [
            dict(id="days", label="Lookback (days)", type="entry", flag="--days",
                 required=False, default="30"),
        ],
    },
    "Classifier": {
        "path": os.path.join(ROOT, "10_classifier", "classifier.py"),
        "desc": "Behavioural clustering — groups index constituents by return/risk/momentum similarity using K-Means, Hierarchical or DBSCAN, with interactive t-SNE / UMAP / PCA visualisation.",
        "note": "💡 Set <b># Clusters</b> to <b>0</b> (default) to auto-select the optimal k via composite silhouette + Calinski-Harabász scan (k = 4–15). Enter any other number to fix k manually. Ignored for DBSCAN.<br><br>Use the <b>Features</b> checkboxes to select which lens drives clustering: <b>Momentum + Risk</b> (default — price-derived, fast) · <b>Value</b> (P/E, P/B, FCF Yield, etc.) · <b>Growth</b> (EPS/Revenue growth, PEG). Enabling Value or Growth features adds ~60s for fundamental data fetching.<br><br><b>DTW + Hierarchical</b> clusters stocks by raw return-series shape similarity using Dynamic Time Warping distance — captures pattern similarity with lag-tolerance that Pearson correlation misses. Feature checkboxes are ignored for DTW.",
        "icon": "🧬",
        "params": [
            dict(id="index",      label="Index",          type="option", flag="--index",
                 required=True,  default="SPX",
                 values=["SPX", "NDX", "DOW", "RUT", "TSX"]),
            dict(id="method",     label="Cluster method", type="option", flag="--method",
                 required=True,  default="kmeans",
                 values=["kmeans", "hierarchical", "dbscan", "dtw"]),
            dict(id="n_clusters", label="# Clusters",     type="entry",  flag="--n_clusters",
                 required=False, default="0",
                 hint="0 = auto (silhouette scan); ignored for DBSCAN"),
            dict(id="viz",        label="2-D embedding",  type="option", flag="--viz",
                 required=True,  default="tsne",
                 values=["tsne", "umap", "pca"]),
            dict(id="features", label="Features", type="checkgroup", flag="--features",
                 required=False,
                 items=[
                     # ── Momentum ──────────────────────────────────────────
                     {"id": "mom_5d",      "label": "Mom 5d",      "group": "Momentum", "default": True},
                     {"id": "mom_10d",     "label": "Mom 10d",     "group": "Momentum", "default": True},
                     {"id": "mom_21d",     "label": "Mom 21d",     "group": "Momentum", "default": True},
                     {"id": "mom_42d",     "label": "Mom 42d",     "group": "Momentum", "default": True},
                     {"id": "mom_63d",     "label": "Mom 63d",     "group": "Momentum", "default": True},
                     {"id": "mom_126d",    "label": "Mom 126d",    "group": "Momentum", "default": True},
                     {"id": "mom_189d",    "label": "Mom 189d",    "group": "Momentum", "default": True},
                     {"id": "mom_252d",    "label": "Mom 252d",    "group": "Momentum", "default": True},
                     # ── Risk ──────────────────────────────────────────────
                     {"id": "vol_21d",           "label": "Vol 21d",        "group": "Risk", "default": True},
                     {"id": "vol_63d",           "label": "Vol 63d",        "group": "Risk", "default": True},
                     {"id": "vol_ratio",         "label": "Vol Ratio",      "group": "Risk", "default": True},
                     {"id": "downside_vol_21d",  "label": "Downside Vol",   "group": "Risk", "default": True},
                     {"id": "sortino_ratio",     "label": "Sortino",        "group": "Risk", "default": True},
                     {"id": "beta",              "label": "Beta",           "group": "Risk", "default": True},
                     {"id": "max_dd_252d",       "label": "Max DD",         "group": "Risk", "default": True},
                     {"id": "autocorr_21d",      "label": "Autocorr",       "group": "Risk", "default": True},
                     {"id": "skew_126d",         "label": "Skewness",       "group": "Risk", "default": True},
                     {"id": "kurtosis_126d",     "label": "Kurtosis",       "group": "Risk", "default": True},
                     {"id": "trend_r2_63d",      "label": "Trend R²",       "group": "Risk", "default": True},
                     {"id": "alpha_252d",        "label": "Alpha",          "group": "Risk", "default": True},
                     {"id": "price_vs_52w_high", "label": "vs 52w High",    "group": "Risk", "default": True},
                     # ── Value (fundamental — opt-in; adds ~60s fetch) ─────
                     {"id": "pe_ratio",   "label": "P/E",          "group": "Value", "default": False},
                     {"id": "pb_ratio",   "label": "P/B",          "group": "Value", "default": False},
                     {"id": "ps_ratio",   "label": "P/S",          "group": "Value", "default": False},
                     {"id": "div_yield",  "label": "Div Yield",    "group": "Value", "default": False},
                     {"id": "fcf_yield",  "label": "FCF Yield",    "group": "Value", "default": False},
                     {"id": "ev_ebitda",  "label": "EV/EBITDA",    "group": "Value", "default": False},
                     {"id": "ev_sales",   "label": "EV/Sales",     "group": "Value", "default": False},
                     # ── Growth (fundamental — opt-in; adds ~60s fetch) ────
                     {"id": "eps_growth_1y", "label": "EPS Growth",   "group": "Growth", "default": False},
                     {"id": "rev_growth_1y", "label": "Rev Growth",   "group": "Growth", "default": False},
                     {"id": "peg_ratio",     "label": "PEG",          "group": "Growth", "default": False},
                     {"id": "gross_margin",  "label": "Gross Margin", "group": "Growth", "default": False},
                     # ── Quality (fundamental — opt-in; adds ~60s fetch) ───
                     {"id": "roe",           "label": "ROE",           "group": "Quality", "default": False},
                     {"id": "roa",           "label": "ROA",           "group": "Quality", "default": False},
                     {"id": "op_margin",     "label": "Op Margin",     "group": "Quality", "default": False},
                     {"id": "net_margin",    "label": "Net Margin",    "group": "Quality", "default": False},
                     {"id": "debt_equity",   "label": "Debt/Equity",   "group": "Quality", "default": False},
                     {"id": "current_ratio", "label": "Current Ratio", "group": "Quality", "default": False},
                 ]),
        ],
    },
    "Magic Formula": {
        "path": os.path.join(ROOT, "12_magicFormula", "magicFormula.py"),
        "desc": "Greenblatt's Magic Formula — dual-ranks stocks by Earnings Yield (EBIT/EV) + Return on Capital (ROIC). Lower combined rank = better blend of cheap + quality.",
        "icon": "📐",
        "params": [
            dict(id="index", label="Index",      type="option", flag="--index",
                 required=True,  default="SPX",  values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",   label="Top N",      type="entry",  flag="--top",
                 required=False, default="",
                 hint="Stocks to screen (default: all index members)"),
            dict(id="csv",   label="Export CSV", type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Quality Screener": {
        "path": os.path.join(ROOT, "13_qualityScreener", "qualityScreener.py"),
        "desc": "Quality / Compounder screener — scores stocks 0-100 across 4 pillars: Return on Capital (35pts), Profitability (25pts), Balance Sheet (25pts), Capital Efficiency (15pts).",
        "icon": "🏆",
        "params": [
            dict(id="index",     label="Index",          type="option", flag="--index",
                 required=True,  default="SPX",          values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",       label="Top N",          type="entry",  flag="--top",
                 required=False, default="",
                 hint="Stocks to screen (default: all index members)"),
            dict(id="min_score", label="Min quality score", type="entry", flag="--min-score",
                 required=False, default="",
                 hint="Only show stocks with score ≥ N (e.g. 70 for B+ and above)"),
            dict(id="csv",       label="Export CSV",     type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "CANSLIM": {
        "path": os.path.join(ROOT, "14_canslim", "canslim.py"),
        "desc": "O'Neil's CANSLIM method — scores stocks on 6 factors: Current EPS growth, Annual earnings quality, Near 52w high, Supply/size, Leader relative strength, Institutional proxy. Score 100 = ideal setup.",
        "icon": "📈",
        "params": [
            dict(id="index",     label="Index",          type="option", flag="--index",
                 required=True,  default="SPX",          values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",       label="Top N",          type="entry",  flag="--top",
                 required=False, default="",
                 hint="Stocks to screen (default: all index members)"),
            dict(id="min_score", label="Min CANSLIM score", type="entry", flag="--min-score",
                 required=False, default="",
                 hint="Only show stocks with score ≥ N (e.g. 60 for B+ and above)"),
            dict(id="csv",       label="Export CSV",     type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Earnings Acceleration": {
        "path": os.path.join(ROOT, "15_earningsAccel", "earningsAccel.py"),
        "desc": "Earnings Acceleration screener — scores stocks on high-magnitude growth + margin quality: EPS growth rate (35pts), Revenue growth (30pts), Gross+Op margin (20pts), ROE/ROIC quality (15pts).",
        "icon": "🚀",
        "params": [
            dict(id="index",     label="Index",          type="option", flag="--index",
                 required=True,  default="SPX",          values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",       label="Top N",          type="entry",  flag="--top",
                 required=False, default="",
                 hint="Stocks to screen (default: all index members)"),
            dict(id="min_score", label="Min accel score", type="entry", flag="--min-score",
                 required=False, default="",
                 hint="Only show stocks with score ≥ N (e.g. 65 for A-grade)"),
            dict(id="csv",       label="Export CSV",     type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Multi-Factor Growth": {
        "path": os.path.join(ROOT, "11_growthScreeners", "growthScreeners.py"),
        "desc": "7 growth screeners in one pass — RS Rating, EPS Acceleration, Estimate Revision, Beat Rate, GARP, FCF Compounder, Gross Margin Expansion.",
        "icon": "🌱",
        "params": [
            dict(id="index", label="Index", type="option", flag="--index",
                 required=False, default="SPX",
                 values=["SPX", "NDX", "DOW", "RUT", "TSX"]),
            dict(id="tickers", label="Tickers (comma-sep, overrides index)",
                 type="entry", flag="--tickers",
                 required=False, default=""),
        ],
    },
    "Topic Sentiment": {
        "path": os.path.join(ROOT, "16_topicSentiment", "topicSentiment.py"),
        "desc": "Multi-source sentiment analysis for any keyword, phrase, or topic — news, Reddit, Hacker News, Google Trends, word clouds & sentiment timeline.",
        "icon": "🌡️",
        "params": [
            dict(id="query", label="Query / keyword", type="entry", flag="--query",
                 required=True, default="",
                 hint='Any topic, phrase, or ticker — e.g. "Federal Reserve" or "NVDA earnings"'),
        ],
    },
    "Price Forecast": {
        "path": os.path.join(ROOT, "11_priceForecast", "priceForecast.py"),
        "desc": "ARIMA + ETS price forecasting with walk-forward backtest — compares model accuracy against a naive random-walk baseline and surfaces uncertainty cones over the forecast horizon.",
        "note": "💡 ARIMA models log-returns (stationary). ETS models log-price levels with a damped trend. Both are benchmarked against the naive random-walk. Outputs include residual ACF diagnostics and a full backtest over the last 252 trading days.",
        "icon": "🔮",
        "params": [
            dict(id="ticker",  label="Ticker",           type="entry",  flag="--ticker",
                 required=True,  default="AAPL",
                 hint="Any yfinance-supported ticker (e.g. AAPL, BTC-USD, ^GSPC)"),
            dict(id="horizon", label="Forecast horizon", type="entry",  flag="--horizon",
                 required=False, default="30",
                 hint="Trading days ahead (5–252, default 30)"),
            dict(id="model",   label="Model",            type="option", flag="--model",
                 required=False, default="both",
                 values=["both", "arima", "ets"]),
            dict(id="period",  label="History",          type="option", flag="--period",
                 required=False, default="5y",
                 values=["2y", "5y", "10y"]),
        ],
    },
    "Return Predictor": {
        "path": os.path.join(ROOT, "17_returnPredictor", "returnPredictor.py"),
        "desc": "XGBoost cross-sectional factor model — trains on momentum, valuation & quality snapshots to rank stocks by predicted forward return. Walk-forward IC reported.",
        "icon": "📈",
        "params": [
            dict(id="csv_file", label="Ticker CSV (overrides index/tickers)",
                 type="file", flag="--csv-file",
                 required=False, default="",
                 hint="CSV with a 'ticker' column and optional 'weight' column"),
            dict(id="index",   label="Index",   type="option", flag="--index",
                 required=False, default="SPX",
                 values=["SPX", "NDX", "DOW", "RUT"]),
            dict(id="tickers", label="Tickers (comma-sep)",
                 type="entry", flag="--tickers",
                 required=False, default=""),
            dict(id="horizon", label="Horizon (trading days)",
                 type="option", flag="--horizon",
                 required=False, default="63",
                 values=["30", "63", "126"]),
            dict(id="top",     label="Top N by market cap", type="entry", flag="--top",
                 required=False, default="150"),
        ],
    },
    "Earnings Predictor": {
        "path": os.path.join(ROOT, "18_earningsPredictor", "earningsPredictor.py"),
        "desc": "XGBoost beat/miss classifier — predicts earnings beat probability for stocks reporting in the next 60 days using beat rate history, analyst revisions & price momentum.",
        "icon": "🎯",
        "params": [
            dict(id="csv_file", label="Ticker CSV (overrides index/tickers)",
                 type="file", flag="--csv-file",
                 required=False, default="",
                 hint="CSV with a 'ticker' column"),
            dict(id="index",   label="Index",   type="option", flag="--index",
                 required=False, default="SPX",
                 values=["SPX", "NDX", "DOW", "RUT"]),
            dict(id="tickers", label="Tickers (comma-sep)",
                 type="entry", flag="--tickers",
                 required=False, default=""),
            dict(id="top",     label="Top N by market cap", type="entry", flag="--top",
                 required=False, default="100"),
            dict(id="days",    label="Look-ahead (days)",   type="entry", flag="--days",
                 required=False, default="60"),
        ],
    },
    "Regime Detector": {
        "path": os.path.join(ROOT, "19_regimeDetector", "regimeDetector.py"),
        "desc": "Gaussian Mixture Model on macro indicators (VIX, yield curve, credit spread, SPY momentum) — classifies the current market regime and recommends WACC / terminal-growth adjustments.",
        "icon": "🧭",
        "params": [
            dict(id="lookback", label="History (years)", type="option", flag="--lookback",
                 required=False, default="15",
                 values=["10", "15", "20"]),
        ],
    },
    "Risk Model": {
        "path": os.path.join(ROOT, "20_riskModel", "riskModel.py"),
        "desc": "GARCH(1,1) conditional volatility forecasts per holding — 30-day VaR/CVaR, rolling beta, Sharpe, max drawdown, correlation heatmap & portfolio-level risk summary.",
        "icon": "🛡️",
        "params": [
            dict(id="tickers",   label="Tickers (comma-sep; blank = Deep Dive list)",
                 type="entry", flag="--tickers",
                 required=False, default="",
                 hint="Leave blank to use your Deep Dive list"),
            dict(id="weights",   label="Weights (comma-sep, must match tickers)",
                 type="entry", flag="--weights",
                 required=False, default="",
                 hint="e.g. 0.4,0.3,0.3 — equal-weight if omitted"),
            dict(id="benchmark", label="Benchmark", type="option", flag="--benchmark",
                 required=False, default="SPY",
                 values=["SPY", "QQQ", "IWM"]),
            dict(id="lookback",  label="History (trading days)", type="option", flag="--lookback",
                 required=False, default="504",
                 values=["252", "504", "756"]),
        ],
    },
    "Breakout Detector": {
        "path": os.path.join(ROOT, "21_breakoutDetector", "breakoutDetector.py"),
        "desc": "XGBoost breakout classifier — ranks stocks by probability of an imminent +8% price move using volume surge, RSI, Bollinger Bands, MA ratios & momentum features.",
        "icon": "🚀",
        "params": [
            dict(id="csv_file",  label="Ticker CSV (overrides index/tickers)",
                 type="file", flag="--csv-file",
                 required=False, default="",
                 hint="CSV with a 'ticker' column and optional 'weight' column"),
            dict(id="index",     label="Index",   type="option", flag="--index",
                 required=False, default="SPX",
                 values=["SPX", "NDX", "DOW", "RUT"]),
            dict(id="tickers",   label="Tickers (comma-sep)",
                 type="entry", flag="--tickers",
                 required=False, default=""),
            dict(id="top",       label="Top N by market cap", type="entry", flag="--top",
                 required=False, default="150"),
            dict(id="threshold", label="Breakout threshold (%)", type="entry", flag="--threshold",
                 required=False, default="8.0"),
        ],
    },
    "Sentiment Lead": {
        "path": os.path.join(ROOT, "22_sentimentLead", "sentimentLead.py"),
        "desc": "Tests whether news sentiment leads price for a ticker — VADER scoring, lead-lag correlation, walk-forward OLS regression & OOS R² signal-quality badge.",
        "icon": "🔗",
        "params": [
            dict(id="ticker",   label="Ticker", type="entry", flag="--ticker",
                 required=True, default="AAPL",
                 hint="Any yfinance-supported ticker"),
            dict(id="lookback", label="Lookback (days)", type="option", flag="--lookback",
                 required=False, default="30",
                 values=["14", "30", "60", "90"]),
            dict(id="lags",        label="Max lag (days)", type="option", flag="--lags",
                 required=False, default="10",
                 values=["5", "10", "20"]),
            dict(id="finnhub_key", label="Finnhub API key (optional)", type="entry",
                 flag="--finnhub-key", required=False, default="",
                 hint="Free key at finnhub.io — unlocks up to 1 year of historical news"),
        ],
    },
}

# Sidebar group layout — defines sections and order shown in the UI
SIDEBAR_GROUPS = [
    {"label": "MACRO TRENDS",      "scripts": ["Macro Dashboard"]},
    {"label": "SCREENERS",         "scripts": ["Value Screener", "Growth Screener",
                                               "Magic Formula", "Quality Screener",
                                               "CANSLIM", "Earnings Acceleration",
                                               "Multi-Factor Growth"]},
    {"label": "VALUATION",         "scripts": ["Valuation Master", "Run Model", "Scatter Plots",
                                               "Price Forecast"]},
    {"label": "PORTFOLIO ANALYSIS","scripts": ["Sentiment Analyzer", "Portfolio Analyzer"]},
    {"label": "WATCHLIST",         "scripts": ["Watchlist Tracker", "Deep Dive List"]},
    {"label": "RESEARCH",          "scripts": ["Research Scanner", "Topic Explorer",
                                               "Topic Sentiment", "Classifier"]},
    {"label": "PREDICTIVE MODELS", "scripts": ["Return Predictor", "Earnings Predictor",
                                               "Regime Detector", "Risk Model",
                                               "Breakout Detector", "Sentiment Lead"]},
]

# ── Active runs ───────────────────────────────────────────────────────────────
# run_id → {"proc": Popen, "q": Queue, "done": bool, "tmpfile": path|None, "cwd": str}
_runs: dict = {}
_runs_lock   = threading.Lock()

import re
# Matches any whitespace-delimited token that is a file path with a known
# output extension.  Handles all scripts:
#   "  Saved: /abs/path/file.html"          ← valuationMaster, portfolioAnalyzer
#   "  HTML saved: relpath/file.html"       ← growthScreener, valueScreener
#   "  ✓  Report saved → /path/file.html  (42 KB)"  ← sentiment, macro
#   "  Plot saved → /abs/path/file.png"    ← run_model.py (matplotlib PNG)
_OUTPUT_PAT = re.compile(r'(\S+\.(?:html|png|jpg|jpeg|svg|pdf))\b', re.IGNORECASE)


def _reader_thread(run_id: str) -> None:
    """Read subprocess stdout into the run's queue; put None sentinel when done.

    Also scans output lines for HTML file paths printed by the script (e.g.
    "  Saved: /path/to/report.html").  When the process finishes successfully,
    opens the last detected HTML file via webbrowser from the Flask server
    process, which has the correct macOS Aqua/window-server context.

    Scripts launched as subprocesses with start_new_session=True may have their
    own webbrowser.open() calls fail silently on macOS — opening from here is
    the reliable fallback.
    """
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        return

    proc        = run["proc"]
    q           = run["q"]
    cwd         = run.get("cwd") or ""
    output_path    = None   # last matched output file seen in the script's stdout
    _last_candidate = None  # last resolved candidate path (for post-wait re-check)

    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
            # Detect output file path from the script's print() statements.
            # Covers HTML reports (all screeners/analyzers) and PNG plots (run_model.py).
            m = _OUTPUT_PAT.search(line)
            if m:
                candidate = m.group(1).rstrip(")>\"'")
                # Resolve relative paths against the script's working directory
                if not os.path.isabs(candidate) and cwd:
                    candidate = os.path.join(cwd, candidate)
                candidate = os.path.abspath(candidate)
                _last_candidate = candidate   # remember even if file not yet visible
                if os.path.isfile(candidate):
                    output_path = candidate

        proc.stdout.close()
        proc.wait()
        rc = proc.returncode
        sep = "─" * 54
        q.put(f"\n{sep}\n")
        q.put("  ✓  Completed (exit 0)\n" if rc == 0 else f"  ✗  Exited with code {rc}\n")
        q.put(f"{sep}\n")

        # Re-check the candidate path now that the process has fully exited and
        # all file writes are guaranteed flushed.  Handles the rare case where
        # os.path.isfile() returned False during the streaming loop (e.g. a brief
        # filesystem flush delay right after plt.savefig()).
        if _last_candidate and not output_path and os.path.isfile(_last_candidate):
            output_path = _last_candidate

        # Open output file.  HTML reports are served through Flask at
        # /view?path=... so the page shares the same origin as the API —
        # this avoids Safari/Chrome blocking file:// → localhost fetches
        # (which broke the "Add to Deep Dive" button).
        # Non-HTML outputs (PNG/PDF) still use the native 'open' command.
        if output_path and rc == 0:
            import platform as _platform
            import urllib.parse as _up
            if output_path.lower().endswith(".html"):
                url = (f"http://localhost:{_SERVER_PORT}/view"
                       f"?path={_up.quote(output_path, safe='')}")
                if _platform.system() == "Darwin":
                    subprocess.Popen(["open", url])
                else:
                    webbrowser.open(url)
            else:
                # Images / PDFs — open with native viewer
                if _platform.system() == "Darwin":
                    subprocess.Popen(["open", output_path])
                else:
                    webbrowser.open("file://" + output_path)

    except Exception as exc:
        q.put(f"\nERROR: {exc}\n")
    finally:
        q.put(None)  # sentinel
        with _runs_lock:
            if run_id in _runs:
                _runs[run_id]["done"] = True
            # Clean up any temp uploaded file
            tmp = _runs.get(run_id, {}).get("tmpfile")
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ── API routes ────────────────────────────────────────────────────────────────

@app.after_request
def _cors(resp):
    """Add CORS + Private Network Access headers to every response.

    Chrome's Private Network Access policy (chrome 94+) requires
    Access-Control-Allow-Private-Network: true when a file:// page fetches
    http://localhost — without it the pre-flight is blocked and the
    'Add to Deep Dive' button silently falls back to a CSV download.
    """
    resp.headers["Access-Control-Allow-Origin"]          = "*"
    resp.headers["Access-Control-Allow-Methods"]         = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"]         = "Content-Type"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


@app.route("/")
def index():
    return HTML


@app.route("/view")
def view_report():
    """Serve a generated HTML report through Flask so it shares the same origin
    as the API — avoids browser blocking of file:// → localhost fetches."""
    import urllib.parse as _up
    path = _up.unquote(request.args.get("path", ""))
    # Security: only serve files that live inside the project directory
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(ROOT)):
        return "Forbidden", 403
    if not os.path.isfile(real) or not real.lower().endswith(".html"):
        return "Not found", 404
    with open(real, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/scripts")
def api_scripts():
    return jsonify(SCRIPTS)


@app.route("/api/groups")
def api_groups():
    return jsonify(SIDEBAR_GROUPS)


@app.route("/api/run", methods=["POST"])
def api_run():
    data        = request.get_json(force=True, silent=True) or {}
    script_name = data.get("script", "")
    params      = data.get("params", {})

    if script_name not in SCRIPTS:
        return jsonify({"error": f"Unknown script: {script_name}"}), 400

    info = SCRIPTS[script_name]
    cmd  = [sys.executable, "-u", info["path"]]  # -u = unbuffered stdout
    cwd  = os.path.dirname(info["path"])

    # Pre-scan: if any param has mutex_with and a non-empty value, skip those peers
    mutex_skip: set[str] = set()
    for p in info["params"]:
        val_pre = str(params.get(p["id"], p.get("default", ""))).strip()
        if val_pre and p.get("mutex_with"):
            mutex_skip.update(p["mutex_with"])

    for p in info["params"]:
        pid   = p["id"]
        if pid in mutex_skip:
            continue
        ptype = p["type"]
        flag  = p.get("flag")
        val   = params.get(pid, p.get("default", ""))

        if ptype == "check":
            if val and flag:
                cmd.append(flag)
        elif ptype == "file":
            val = str(val).strip()
            if val:
                if flag:
                    cmd.extend([flag, val])
                else:
                    cmd.append(val)
        else:
            val = str(val).strip()
            if not val:
                continue
            if flag:
                cmd.extend([flag, val])
            else:
                cmd.append(val)

    run_id = uuid.uuid4().hex[:8]
    # Pass VALUATION_SUITE_LAUNCHED so scripts skip their own webbrowser.open()
    # calls — _reader_thread is the single place that opens output files.
    # MPLBACKEND=Agg forces matplotlib to use the file-only Agg backend for any
    # subprocess that generates plots.  Without this, macOS defaults to the
    # MacOSX interactive backend, which requires a window-server connection that
    # start_new_session=True subprocesses don't have — plt.savefig() silently
    # produces nothing.  Setting it here is more reliable than calling
    # matplotlib.use("Agg") inside the script because it takes effect before
    # any import can touch the backend.
    child_env = {**os.environ, "VALUATION_SUITE_LAUNCHED": "1",
                 "MPLBACKEND": "Agg",
                 "VALUATION_SUITE_PORT": str(_SERVER_PORT)}
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            start_new_session=True,
            env=child_env,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    q = queue.Queue()
    with _runs_lock:
        _runs[run_id] = {"proc": proc, "q": q, "done": False, "tmpfile": None, "cwd": cwd}

    threading.Thread(target=_reader_thread, args=(run_id,), daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/api/stream/<run_id>")
def api_stream(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        return Response("data: [DONE]\n\n", mimetype="text/event-stream")

    def generate():
        q = run["q"]
        while True:
            try:
                line = q.get(timeout=25)
            except queue.Empty:
                yield ": keep-alive\n\n"  # prevent proxy timeouts
                continue
            if line is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(line)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stop/<run_id>", methods=["POST"])
def api_stop(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if run:
        proc = run["proc"]
        if proc.poll() is None:
            proc.terminate()
    return jsonify({"ok": True})


@app.route("/api/exit", methods=["POST"])
def api_exit():
    # os._exit() terminates the entire process regardless of which thread calls
    # it.  sys.exit() only raises SystemExit in the calling thread — when called
    # from a Timer thread it leaves the Flask main thread (and the bound socket)
    # alive, so the port stays occupied and macOS keeps the app registered.
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


@app.route("/api/watchlist/add", methods=["POST", "OPTIONS"])
def api_watchlist_add():
    """Append checked tickers to deepDiveTickers/deepDiveTickers.csv (append-only, deduped).
    Supports CORS so the endpoint is reachable from file:// HTML reports."""
    # Pre-flight — CORS/PNA headers are added by the global @after_request hook
    if request.method == "OPTIONS":
        return make_response("", 204)

    data        = request.get_json(force=True, silent=True) or {}
    new_tickers = [t.upper().strip() for t in data.get("tickers", []) if str(t).strip()]

    # Load existing tickers so we can deduplicate
    existing: set[str] = set()
    if os.path.exists(DEEPDIVE_CSV):
        with open(DEEPDIVE_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = (row.get("ticker") or row.get("symbol") or "").upper().strip()
                if t:
                    existing.add(t)

    to_add  = [t for t in new_tickers if t and t not in existing]
    skipped = len(new_tickers) - len(to_add)
    today   = datetime.now().strftime("%Y-%m-%d")

    if to_add:
        write_header = not os.path.exists(DEEPDIVE_CSV)
        with open(DEEPDIVE_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ticker", "shares", "added_date"])
            for t in to_add:
                w.writerow([t, 0, today])

    return jsonify({"added": len(to_add), "skipped": skipped,
                    "total": len(existing) + len(to_add)})


@app.route("/api/deep-dive/list")
def api_deep_dive_list():
    """Return current deep dive tickers as JSON list."""
    tickers = []
    if os.path.exists(DEEPDIVE_CSV):
        with open(DEEPDIVE_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = (row.get("ticker") or row.get("symbol") or "").upper().strip()
                d = row.get("added_date", "")
                if t:
                    tickers.append({"ticker": t, "added_date": d})
    return jsonify(tickers)


@app.route("/api/deep-dive/remove", methods=["POST"])
def api_deep_dive_remove():
    """Remove a single ticker from the deep dive CSV."""
    data   = request.get_json(force=True, silent=True) or {}
    target = (data.get("ticker") or "").upper().strip()
    if not target:
        return jsonify({"error": "No ticker provided"}), 400

    rows = []
    if os.path.exists(DEEPDIVE_CSV):
        with open(DEEPDIVE_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    kept    = [r for r in rows if (r.get("ticker") or "").upper().strip() != target]
    removed = len(rows) - len(kept)

    with open(DEEPDIVE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "shares", "added_date"])
        w.writeheader()
        w.writerows(kept)

    return jsonify({"removed": removed, "total": len(kept)})


@app.route("/deep-dive")
def deep_dive_page():
    """Interactive editor for deepDiveTickers.csv."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep Dive List</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #161b2e; color: #e8eaf0;
         font-family: 'Inter', system-ui, sans-serif; font-size: 13px;
         padding: 32px; min-height: 100vh; }
  h1   { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
  .sub { color: #6b7194; font-size: 12px; margin-bottom: 28px; }
  .card { background: #1e2538; border: 1px solid #252a3a;
          border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  /* Add bar */
  .add-row { display: flex; gap: 10px; align-items: center; }
  .add-row input { flex: 1; background: #252a3a; border: 1px solid #2e3550;
                   border-radius: 6px; padding: 8px 12px; color: #e8eaf0;
                   font-size: 13px; outline: none; }
  .add-row input:focus { border-color: #4f8ef7; }
  .btn-add { background: #4f8ef7; color: #fff; border: none; border-radius: 6px;
             padding: 8px 18px; font-size: 12px; font-weight: 600; cursor: pointer;
             white-space: nowrap; }
  .btn-add:hover { background: #6ba3ff; }
  .hint { color: #6b7194; font-size: 11px; margin-top: 8px; }
  /* Ticker list */
  .ticker-grid { display: flex; flex-wrap: wrap; gap: 8px; min-height: 48px; }
  .ticker-chip { display: inline-flex; align-items: center; gap: 6px;
                 background: #252a3a; border: 1px solid #2e3550;
                 border-radius: 20px; padding: 5px 12px 5px 14px;
                 font-size: 12px; font-weight: 600; }
  .ticker-chip .date { color: #6b7194; font-size: 10px; font-weight: 400; margin-left: 2px; }
  .btn-rm { background: none; border: none; color: #6b7194; cursor: pointer;
            font-size: 14px; line-height: 1; padding: 0 2px; border-radius: 50%; }
  .btn-rm:hover { color: #e05c5c; background: #3a2020; }
  .empty { color: #6b7194; font-style: italic; font-size: 12px; padding: 8px 0; }
  /* Stats bar */
  .stats { display: flex; gap: 24px; padding: 14px 20px;
           background: #1a1f35; border: 1px solid #252a3a;
           border-radius: 8px; margin-bottom: 16px; }
  .stat-val { font-size: 20px; font-weight: 700; }
  .stat-lbl { font-size: 11px; color: #6b7194; margin-top: 2px; }
  /* Toast */
  #toast { position: fixed; bottom: 24px; right: 24px; background: #1e2538;
           border: 1px solid #252a3a; color: #e8eaf0; padding: 10px 16px;
           border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity .3s;
           z-index: 999; max-width: 300px; }
  #toast.show { opacity: 1; }
  #toast.warn { border-color: #f0a500; color: #f0a500; }
  #toast.err  { border-color: #e05c5c; color: #e05c5c; }
  .section-label { font-size: 11px; font-weight: 700; color: #6b7194;
                   text-transform: uppercase; letter-spacing: .06em;
                   margin-bottom: 12px; }
  .copy-btn { background: transparent; border: 1px solid #2e3550; color: #6b7194;
              border-radius: 5px; padding: 4px 10px; font-size: 11px; cursor: pointer;
              margin-left: auto; }
  .copy-btn:hover { color: #e8eaf0; border-color: #4f8ef7; }
  .card-header { display: flex; align-items: center; margin-bottom: 14px; }
</style>
</head>
<body>
<h1>🔬  Deep Dive List</h1>
<p class="sub">Tickers flagged from screeners for deeper analysis. Saved to <code>deepDiveTickers/deepDiveTickers.csv</code>.</p>

<div class="stats">
  <div><div class="stat-val" id="stat-count">—</div><div class="stat-lbl">Tickers</div></div>
  <div><div class="stat-val" id="stat-latest">—</div><div class="stat-lbl">Last Added</div></div>
</div>

<div class="card">
  <div class="section-label">Add Tickers</div>
  <div class="add-row">
    <input id="add-input" type="text" placeholder="AAPL, MSFT, NVDA  (comma-separated)" autocomplete="off">
    <button class="btn-add" id="add-btn">+ Add</button>
  </div>
  <p class="hint">Press Enter or click Add. Duplicates are ignored.</p>
</div>

<div class="card">
  <div class="card-header">
    <span class="section-label" style="margin-bottom:0">Current List</span>
    <button class="copy-btn" id="copy-btn" title="Copy all tickers as comma-separated list">Copy tickers</button>
  </div>
  <div class="ticker-grid" id="ticker-grid">
    <span class="empty">Loading…</span>
  </div>
</div>

<div id="toast"></div>

<script>
const BASE = window.location.origin;

function toast(msg, type) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'show' + (type ? ' ' + type : '');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.className = '', 3200);
}

async function loadList() {
  const res  = await fetch(BASE + '/api/deep-dive/list');
  const data = await res.json();
  const grid = document.getElementById('ticker-grid');

  document.getElementById('stat-count').textContent = data.length;
  document.getElementById('stat-latest').textContent =
    data.length ? data[data.length - 1].added_date || '—' : '—';

  if (!data.length) {
    grid.innerHTML = '<span class="empty">No tickers yet — add some above.</span>';
    return;
  }

  grid.innerHTML = '';
  data.forEach(({ticker, added_date}) => {
    const chip = document.createElement('div');
    chip.className = 'ticker-chip';
    chip.id = 'chip-' + ticker;
    chip.innerHTML =
      ticker +
      (added_date ? '<span class="date">' + added_date + '</span>' : '') +
      '<button class="btn-rm" data-ticker="' + ticker + '" title="Remove ' + ticker + '">&times;</button>';
    grid.appendChild(chip);
  });
}

async function removeTicker(ticker) {
  const chip = document.getElementById('chip-' + ticker);
  if (chip) chip.style.opacity = '0.4';
  const res  = await fetch(BASE + '/api/deep-dive/remove', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ticker})
  });
  const data = await res.json();
  if (data.removed) {
    toast('\u2713 ' + ticker + ' removed  (' + data.total + ' remaining)');
    loadList();
  } else {
    if (chip) chip.style.opacity = '1';
    toast('Could not remove ' + ticker, 'err');
  }
}

async function addTickers() {
  const raw     = document.getElementById('add-input').value.trim();
  if (!raw) return;
  const tickers = raw.split(',').map(t => t.trim().toUpperCase()).filter(Boolean);
  if (!tickers.length) return;

  const res  = await fetch(BASE + '/api/watchlist/add', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tickers})
  });
  const data = await res.json();
  document.getElementById('add-input').value = '';
  toast('\u2713 ' + data.added + ' added' +
        (data.skipped ? '  \u00b7  ' + data.skipped + ' already present' : '') +
        '  (' + data.total + ' total)');
  loadList();
}

// Events
document.getElementById('add-btn').addEventListener('click', addTickers);
document.getElementById('add-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') addTickers();
});
document.getElementById('ticker-grid').addEventListener('click', e => {
  const btn = e.target.closest('.btn-rm');
  if (btn) removeTicker(btn.dataset.ticker);
});
document.getElementById('copy-btn').addEventListener('click', () => {
  const chips = [...document.querySelectorAll('.ticker-chip')];
  const text  = chips.map(c => c.id.replace('chip-', '')).join(', ');
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => toast('\u2713 Copied to clipboard'));
});

loadList();
</script>
</body>
</html>"""


@app.route("/run-prefill")
def run_prefill():
    """
    Redirect to main page with JS that pre-selects a script and pre-fills a ticker.
    Usage: /run-prefill?script=Valuation+Master&ticker=AAPL
    """
    script = request.args.get("script", "")
    ticker = request.args.get("ticker", "")
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Loading...</title>
<script>
  const script = "{s}";
  const ticker = "{t}";
  sessionStorage.setItem("prefill_script", script);
  sessionStorage.setItem("prefill_ticker", ticker);
  window.location.href = "/";
</script></head>
<body style="background:#0d0f14;color:#e8eaf2;font-family:monospace;padding:40px;">
  Redirecting to {s} for {t}...
</body></html>""".format(s=script.replace('"', ""), t=ticker.replace('"', ""))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Save an uploaded CSV.  Optional ?script= query param routes to the correct directory."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received"}), 400
    safe_name = os.path.basename(f.filename)
    script    = request.args.get("script", "")
    # All CSV uploads land in one shared folder so the same file works across
    # Portfolio Analyzer, Sentiment Analyzer, and Scatter Plots without re-uploading.
    dest_dir = os.path.join(ROOT, "4_portfolioAnalyzer")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, safe_name)
    f.save(dest)
    return jsonify({"path": dest})


# ── Live quote cache (60-second TTL) ─────────────────────────────────────────
_lq_cache: dict = {"ts": 0.0, "data": {}}


@app.route("/api/live-quotes")
def api_live_quotes():
    """Return live 1-day % change for a comma-separated list of tickers.

    Used by the macro dashboard HTML to populate the live '1D' column.
    Results are cached for 60 seconds so rapid page reloads don't hammer yfinance.

    Query params:
        symbols  — comma-separated ticker list, e.g. SPY,QQQ,TLT,BTC-USD
    Returns:
        JSON object  {ticker: changesPercentage | null, ...}
    """
    import time

    symbols_raw = request.args.get("symbols", "")
    tickers = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    if not tickers:
        return _cors(jsonify({}))

    now = time.time()
    if now - _lq_cache["ts"] < 60:
        data = {t: _lq_cache["data"].get(t) for t in tickers}
        return _cors(jsonify(data))

    # Fetch fresh data via yfinance batch download
    result: dict = {}
    try:
        import yfinance as yf
        raw = yf.download(
            tickers, period="5d", interval="1d",
            auto_adjust=True, progress=False,
        )
        # raw['Close'] is a DataFrame (tickers as cols) for multi-ticker downloads;
        # for a single ticker it may be a plain Series — normalise to DataFrame.
        closes = raw.get("Close", raw)
        if hasattr(closes, "to_frame"):          # single-ticker Series
            closes = closes.to_frame(name=tickers[0])
        for t in tickers:
            try:
                if t not in closes.columns:
                    result[t] = None
                    continue
                col = closes[t].dropna()
                if len(col) >= 2:
                    prev, curr = float(col.iloc[-2]), float(col.iloc[-1])
                    result[t] = round((curr - prev) / prev * 100, 2) if prev else None
                else:
                    result[t] = None
            except Exception:
                result[t] = None
    except Exception:
        pass

    _lq_cache["ts"]   = time.time()
    _lq_cache["data"] = result

    return _cors(jsonify({t: result.get(t) for t in tickers}))


# ── Topic Explorer page (served from Flask so AJAX same-origin works) ─────────

@app.route("/topic-explorer")
def topic_explorer_page():
    """Serve the Topic Explorer interactive HTML from Flask.

    Serving it via Flask (http://127.0.0.1:5050/topic-explorer) instead of as
    a file:// URL means all AJAX calls to /api/research-topics are same-origin
    and are never blocked by the browser's CORS policy.
    """
    scanner_dir = os.path.join(ROOT, "9_researchScanner")
    if scanner_dir not in sys.path:
        sys.path.insert(0, scanner_dir)
    try:
        from topicExplorer import build_html  # noqa: PLC0415
    except Exception as exc:
        return f"<pre>Error loading topicExplorer: {exc}</pre>", 500

    resp = make_response(build_html())
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


# ── Research Topics API ───────────────────────────────────────────────────────

@app.route("/api/research-topics", methods=["POST", "OPTIONS"])
def api_research_topics():
    """AJAX endpoint used by the Topic Explorer HTML interface.

    Accepts JSON body:
        {
            "topics":  ["ai_ml", "genomics", ...],
            "days":    30,
            "sources": ["arxiv", "nih", "trials"]
        }

    Returns JSON:
        {
            "arxiv":  [...],
            "nih":    [...],
            "trials": [...]
        }
    """
    # CORS preflight
    if request.method == "OPTIONS":
        resp = Response()
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    # Lazy import so topicExplorer.py is only loaded when this endpoint is hit
    scanner_dir = os.path.join(ROOT, "9_researchScanner")
    if scanner_dir not in sys.path:
        sys.path.insert(0, scanner_dir)

    try:
        from topicExplorer import fetch_topics  # noqa: PLC0415
    except Exception as exc:
        return _cors(jsonify({"error": f"Could not import topicExplorer: {exc}"})), 500

    data    = request.get_json(force=True, silent=True) or {}
    topics  = data.get("topics", [])
    days    = int(data.get("days", 30))
    sources = data.get("sources", ["arxiv", "nih", "trials"])

    try:
        results = fetch_topics(topics, days=days, sources=sources)
    except Exception as exc:
        return _cors(jsonify({"error": str(exc)})), 500

    return _cors(jsonify(results))


# ── Embedded HTML / CSS / JS ───────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Valuation Suite</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0e1117;
    --panel:   #161b22;
    --card:    #1c2128;
    --border:  #30363d;
    --accent:  #4f8ef7;
    --green:   #00c896;
    --muted:   #8b949e;
    --text:    #e6edf3;
    --btn-run: #1f6feb;
    --btn-stp: #da3633;
    --console: #060d17;
    --yellow:  #f0a500;
    --red:     #f85149;
    --radius:  8px;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text);
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

  /* ── Layout ── */
  .shell   { display: flex; flex-direction: column; height: 100vh; }
  .topbar  { display: flex; align-items: center; gap: 10px;
             background: var(--panel); border-bottom: 1px solid var(--border);
             padding: 0 20px; height: 52px; flex-shrink: 0; }
  .topbar h1 { font-size: 17px; font-weight: 700; color: var(--text); flex: 1; }
  .btn-exit { font-size: 12px; font-weight: 600; padding: 6px 16px; border: 1px solid var(--border);
              border-radius: 6px; background: transparent; color: var(--muted); cursor: pointer;
              transition: background .15s, color .15s, border-color .15s; }
  .btn-exit:hover { background: var(--btn-stp); border-color: var(--btn-stp); color: #fff; }
  .body    { display: flex; flex: 1; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar { width: 190px; background: var(--panel);
             border-right: 1px solid var(--border);
             display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto; }
  .sidebar-label { font-size: 10px; font-weight: 700; color: var(--muted);
                   letter-spacing: .08em; padding: 18px 16px 6px; }
  .sidebar hr { border: none; border-top: 1px solid var(--border); margin: 0 16px 8px; }
  .sbtn { display: block; width: 100%; text-align: left;
          background: transparent; border: none; color: var(--muted);
          font-size: 13px; padding: 9px 14px; cursor: pointer; border-radius: 6px;
          margin: 1px 6px; width: calc(100% - 12px); transition: background .15s, color .15s; }
  .sbtn:hover   { background: var(--card); color: var(--text); }
  .sbtn.active  { background: var(--card); color: var(--text); font-weight: 600; }

  /* ── Content ── */
  .content { flex: 1; display: flex; flex-direction: column; overflow: hidden; padding: 14px; gap: 10px; }

  /* ── Params card ── */
  .params-card { background: var(--card); border: 1px solid var(--border);
                 border-radius: var(--radius); padding: 14px 18px 10px; flex-shrink: 0; }
  .params-card h2 { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: var(--text); }
  .params-grid { display: flex; flex-wrap: wrap; gap: 10px 24px; align-items: center; }
  .param-group { display: flex; align-items: center; gap: 8px; }
  .param-group label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .param-group label.req { color: var(--text); }
  #params-note { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border);
                 font-size: 11px; color: var(--muted); line-height: 1.5; }

  input[type="text"] {
    background: #0d1117; border: 1px solid var(--border); color: var(--text);
    font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
    padding: 5px 9px; border-radius: 5px; width: 130px; outline: none;
    transition: border-color .15s;
  }
  input[type="text"]:focus { border-color: var(--accent); }

  select {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
    padding: 5px 9px; border-radius: 5px; outline: none; cursor: pointer;
  }

  .toggle { font-size: 12px; padding: 5px 12px; border-radius: 5px;
            border: none; cursor: pointer; font-weight: 600; transition: background .15s; }
  .toggle.on  { background: var(--accent); color: #fff; }
  .toggle.off { background: var(--border); color: var(--muted); }

  /* ── Checkgroup (feature selector) ── */
  .checkgroup-wrap { display: flex; flex-direction: column; gap: 6px; }
  .checkgroup-header { display: flex; align-items: center; gap: 8px; }
  .checkgroup-header .cg-lbl { font-size: 11px; color: var(--muted); text-transform: uppercase;
                                letter-spacing: .05em; }
  .checkgroup-header button { font-size: 10px; padding: 2px 7px; border-radius: 4px;
                               border: 1px solid var(--border); background: transparent;
                               color: var(--muted); cursor: pointer; }
  .checkgroup-header button:hover { color: var(--text); border-color: var(--accent); }
  .checkgroup-grid { display: flex; flex-wrap: wrap; gap: 5px 10px; }
  .checkgroup-item { display: flex; align-items: center; gap: 4px; white-space: nowrap;
                     font-size: 12px; color: var(--text); cursor: pointer; }
  .checkgroup-item input[type="checkbox"] { accent-color: var(--accent); cursor: pointer;
                                            width: 13px; height: 13px; }
  /* Group section headers within a checkgroup (Momentum / Risk / Value / Growth) */
  .checkgroup-section { width: 100%; margin-top: 8px; padding-top: 6px;
                        border-top: 1px solid var(--border);
                        font-size: 10px; color: var(--accent);
                        text-transform: uppercase; letter-spacing: .08em; }
  .checkgroup-section:first-child { margin-top: 0; padding-top: 0; border-top: none; }

  .file-row { display: flex; align-items: center; gap: 8px; }
  .file-row input[type="text"] { width: 280px; }
  .file-row input[type="file"] { display: none; }
  .browse-btn { font-size: 12px; padding: 5px 12px; border-radius: 5px;
                border: 1px solid var(--border); background: var(--border);
                color: var(--text); cursor: pointer; white-space: nowrap; }
  .browse-btn:hover { background: var(--accent); border-color: var(--accent); }

  /* ── Description ── */
  .desc { font-size: 12px; color: var(--muted); flex-shrink: 0; }

  /* ── Button row ── */
  .btn-row { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .btn { font-size: 13px; font-weight: 700; padding: 8px 22px; border: none;
         border-radius: 6px; cursor: pointer; transition: opacity .15s; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-run { background: var(--btn-run); color: #fff; }
  .btn-run:not(:disabled):hover { background: #388bfd; }
  .btn-stp { background: var(--btn-stp); color: #fff; }
  .btn-stp:not(:disabled):hover { background: var(--red); }
  .btn-clr { background: var(--card); color: var(--muted); font-weight: 400; margin-left: auto; }
  .btn-clr:hover { background: var(--border); color: var(--text); }

  /* ── Console ── */
  .console-wrap { flex: 1; background: var(--card); border: 1px solid var(--border);
                  border-radius: var(--radius); display: flex; flex-direction: column;
                  overflow: hidden; min-height: 0; }
  .console-label { font-size: 10px; font-weight: 700; color: var(--muted);
                   letter-spacing: .08em; padding: 8px 12px 4px; flex-shrink: 0; }
  #console { flex: 1; background: var(--console); color: var(--green); overflow-y: auto;
             font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
             line-height: 1.55; padding: 10px 14px; white-space: pre-wrap; word-break: break-word; }

  .c-err  { color: var(--red); }
  .c-warn { color: var(--yellow); }
  .c-ok   { color: var(--green); }
  .c-info { color: var(--accent); }
  .c-hdr  { color: var(--text); font-weight: bold; }
  .c-dim  { color: var(--muted); }
</style>
</head>
<body>
<div class="shell">

  <!-- Top bar -->
  <div class="topbar">
    <h1>📊&nbsp; Valuation Suite</h1>
    <button class="btn-exit" onclick="exitApp()">Exit</button>
  </div>

  <div class="body">

    <!-- Sidebar -->
    <nav class="sidebar" id="sidebar"></nav>

    <!-- Main content -->
    <div class="content">
      <div class="params-card">
        <h2 id="script-title">—</h2>
        <div class="params-grid" id="params-grid"></div>
        <div id="params-note" style="display:none"></div>
      </div>

      <p class="desc" id="script-desc"></p>

      <div class="btn-row">
        <button class="btn btn-run" id="btn-run" onclick="runScript()">▶&nbsp; Run</button>
        <button class="btn btn-stp" id="btn-stp" onclick="stopScript()" disabled>■&nbsp; Stop</button>
        <button class="btn btn-clr" onclick="clearConsole()">Clear</button>
      </div>

      <div class="console-wrap">
        <div class="console-label">OUTPUT</div>
        <div id="console"></div>
      </div>
    </div>
  </div>
</div>

<script>
let SCRIPTS = {};
let currentScript = null;
let currentRunId  = null;
let currentES     = null;   // EventSource

// ── Bootstrap ──────────────────────────────────────────────────────────────
Promise.all([
  fetch("/api/scripts").then(r => r.json()),
  fetch("/api/groups").then(r => r.json()),
]).then(([scripts, groups]) => {
  SCRIPTS = scripts;
  const sidebar = document.getElementById("sidebar");

  groups.forEach((group, gi) => {
    if (gi > 0) {
      const hr = document.createElement("hr");
      sidebar.appendChild(hr);
    }
    const lbl = document.createElement("div");
    lbl.className = "sidebar-label";
    lbl.textContent = group.label;
    sidebar.appendChild(lbl);

    group.scripts.forEach(name => {
      if (!SCRIPTS[name]) return;
      const b = document.createElement("button");
      b.className   = "sbtn";
      b.textContent = SCRIPTS[name].icon + "  " + name;
      b.id          = "sbtn-" + name;
      if (SCRIPTS[name].url) {
        b.onclick = () => window.open(SCRIPTS[name].url, "_blank");
      } else {
        b.onclick = () => selectScript(name);
      }
      sidebar.appendChild(b);
    });
  });

  // Check for prefill from /run-prefill redirect
  const _pScript = sessionStorage.getItem("prefill_script");
  const _pTicker = sessionStorage.getItem("prefill_ticker");
  if (_pScript && SCRIPTS[_pScript]) {
    sessionStorage.removeItem("prefill_script");
    sessionStorage.removeItem("prefill_ticker");
    selectScript(_pScript);
    if (_pTicker) {
      // Find the first text input param and fill it
      setTimeout(() => {
        const inp = document.querySelector("#params-grid input[type=text]");
        if (inp) { inp.value = _pTicker; inp.dispatchEvent(new Event("input")); }
      }, 100);
    }
  } else {
    // Select first script of first group by default
    const firstName = groups[0]?.scripts[0];
    if (firstName && SCRIPTS[firstName]) selectScript(firstName);
  }
});

// ── Script selection ───────────────────────────────────────────────────────
function selectScript(name) {
  currentScript = name;
  const info = SCRIPTS[name];

  document.querySelectorAll(".sbtn").forEach(b => b.classList.remove("active"));
  const activeBtn = document.getElementById("sbtn-" + name);
  if (activeBtn) activeBtn.classList.add("active");

  document.getElementById("script-title").textContent = info.icon + "  " + name;
  document.getElementById("script-desc").textContent  = info.desc;

  const grid = document.getElementById("params-grid");
  grid.innerHTML = "";

  info.params.forEach(p => {
    const group = document.createElement("div");
    group.className = "param-group";

    const lbl = document.createElement("label");
    lbl.textContent = p.label + (p.required ? " *" : "");
    if (p.required) lbl.classList.add("req");
    group.appendChild(lbl);

    if (p.type === "entry") {
      const inp = document.createElement("input");
      inp.type  = "text";
      inp.id    = "param-" + p.id;
      inp.value = p.default || "";
      group.appendChild(inp);

    } else if (p.type === "option") {
      const sel = document.createElement("select");
      sel.id = "param-" + p.id;
      (p.values || []).forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        if (v === p.default) opt.selected = true;
        sel.appendChild(opt);
      });
      group.appendChild(sel);

    } else if (p.type === "check") {
      const btn = document.createElement("button");
      btn.className = "toggle " + (p.default ? "on" : "off");
      btn.id        = "param-" + p.id;
      btn.textContent = p.default ? "● ON" : "○ OFF";
      btn.dataset.val = p.default ? "1" : "0";
      btn.onclick = () => {
        const on = btn.dataset.val === "0";
        btn.dataset.val   = on ? "1" : "0";
        btn.className     = "toggle " + (on ? "on" : "off");
        btn.textContent   = on ? "● ON" : "○ OFF";
      };
      group.appendChild(btn);

    } else if (p.type === "checkgroup") {
      // Override group layout: span full width, stack vertically
      group.style.flexDirection = "column";
      group.style.alignItems    = "flex-start";
      group.style.width         = "100%";

      const wrap = document.createElement("div");
      wrap.className = "checkgroup-wrap";

      // Header row: label + Select All / None buttons
      const hdr = document.createElement("div");
      hdr.className = "checkgroup-header";
      const cgLbl = document.createElement("span");
      cgLbl.className = "cg-lbl";
      cgLbl.textContent = p.label;
      const allBtn  = document.createElement("button");
      allBtn.textContent = "Select all";
      allBtn.onclick = (e) => {
        e.preventDefault();
        wrap.querySelectorAll("input[type='checkbox']").forEach(cb => cb.checked = true);
      };
      const noneBtn = document.createElement("button");
      noneBtn.textContent = "Deselect all";
      noneBtn.onclick = (e) => {
        e.preventDefault();
        wrap.querySelectorAll("input[type='checkbox']").forEach(cb => cb.checked = false);
      };
      hdr.appendChild(cgLbl);
      hdr.appendChild(allBtn);
      hdr.appendChild(noneBtn);
      wrap.appendChild(hdr);

      // Checkbox grid — supports optional item.group for section headers
      const cgGrid = document.createElement("div");
      cgGrid.className = "checkgroup-grid";
      let _lastGroup = null;
      (p.items || []).forEach(item => {
        // Render a section header whenever the group label changes
        const grp = item.group || "";
        if (grp && grp !== _lastGroup) {
          const sec = document.createElement("div");
          sec.className   = "checkgroup-section";
          sec.textContent = grp;
          cgGrid.appendChild(sec);
          _lastGroup = grp;
        }
        const itemWrap = document.createElement("label");
        itemWrap.className = "checkgroup-item";
        const cb = document.createElement("input");
        cb.type    = "checkbox";
        cb.id      = "param-" + p.id + "-" + item.id;
        cb.checked = item.default !== false;
        const span = document.createElement("span");
        span.textContent = item.label;
        itemWrap.appendChild(cb);
        itemWrap.appendChild(span);
        cgGrid.appendChild(itemWrap);
      });
      wrap.appendChild(cgGrid);

      // Remove the auto-generated label (we render our own in the header)
      if (lbl.parentNode === group) group.removeChild(lbl);
      group.appendChild(wrap);

    } else if (p.type === "file") {
      const row = document.createElement("div");
      row.className = "file-row";

      const inp = document.createElement("input");
      inp.type  = "text";
      inp.id    = "param-" + p.id;
      inp.value = p.default || "";

      const fileInp = document.createElement("input");
      fileInp.type  = "file";
      fileInp.accept = ".csv";
      fileInp.id    = "fileinput-" + p.id;
      fileInp.onchange = () => {
        const file = fileInp.files[0];
        if (!file) return;
        const fd = new FormData();
        fd.append("file", file);
        fetch("/api/upload?script=" + encodeURIComponent(currentScript), { method: "POST", body: fd })
          .then(r => r.json())
          .then(data => {
            if (data.path) inp.value = data.path;
            else logLine("  ⚠  Upload failed: " + (data.error || "unknown"), "warn");
          });
      };

      const browseBtn = document.createElement("button");
      browseBtn.className   = "browse-btn";
      browseBtn.textContent = "Browse…";
      browseBtn.onclick     = () => fileInp.click();

      row.appendChild(inp);
      row.appendChild(fileInp);
      row.appendChild(browseBtn);
      group.appendChild(row);
    }

    grid.appendChild(group);
  });

  // Render optional script-level note below the params grid
  const noteEl = document.getElementById("params-note");
  if (noteEl) {
    if (info.note) {
      noteEl.innerHTML      = info.note;
      noteEl.style.display  = "";
    } else {
      noteEl.innerHTML      = "";
      noteEl.style.display  = "none";
    }
  }
}

// ── Build params dict from current form ───────────────────────────────────
function collectParams() {
  const info   = SCRIPTS[currentScript];
  const params = {};
  for (const p of info.params) {
    if (p.type === "check") {
      const btn = document.getElementById("param-" + p.id);
      params[p.id] = btn ? btn.dataset.val === "1" : false;
    } else if (p.type === "checkgroup") {
      const checked = (p.items || [])
        .filter(item => {
          const cb = document.getElementById("param-" + p.id + "-" + item.id);
          return cb ? cb.checked : item.default !== false;
        })
        .map(item => item.id);
      params[p.id] = checked.join(",");
    } else {
      const el = document.getElementById("param-" + p.id);
      params[p.id] = el ? el.value : "";
    }
  }
  return params;
}

// ── Run ────────────────────────────────────────────────────────────────────
function runScript() {
  if (currentRunId) return;

  const info   = SCRIPTS[currentScript];
  const params = collectParams();

  // Validate required
  for (const p of info.params) {
    if (p.required) {
      const v = params[p.id];
      if (!v || !String(v).trim()) {
        logLine("  ⚠  '" + p.label + "' is required.\n", "warn");
        return;
      }
    }
  }

  clearConsole();
  const ts = new Date().toLocaleTimeString();
  logLine("─".repeat(54) + "\n", "dim");
  logLine("  " + info.icon + "   " + currentScript + "   ·   " + ts + "\n", "hdr");
  logLine("─".repeat(54) + "\n\n", "dim");

  setRunning(true);

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ script: currentScript, params }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      logLine("  ERROR: " + data.error + "\n", "err");
      setRunning(false);
      return;
    }
    currentRunId = data.run_id;
    startStream(currentRunId);
  })
  .catch(err => {
    logLine("  ERROR: " + err + "\n", "err");
    setRunning(false);
  });
}

// ── SSE stream ─────────────────────────────────────────────────────────────
function startStream(runId) {
  if (currentES) { currentES.close(); currentES = null; }
  currentES = new EventSource("/api/stream/" + runId);
  currentES.onmessage = e => {
    if (e.data === "[DONE]") {
      currentES.close();
      currentES   = null;
      currentRunId = null;
      setRunning(false);
      return;
    }
    const line = JSON.parse(e.data);
    logLine(line, tagLine(line));
  };
  currentES.onerror = () => {
    currentES.close();
    currentES   = null;
    currentRunId = null;
    setRunning(false);
  };
}

// ── Stop ───────────────────────────────────────────────────────────────────
function stopScript() {
  if (!currentRunId) return;
  fetch("/api/stop/" + currentRunId, { method: "POST" });
  if (currentES) { currentES.close(); currentES = null; }
  logLine("\n  [Stopped by user]\n", "warn");
  currentRunId = null;
  setRunning(false);
}

// ── Button state ───────────────────────────────────────────────────────────
function setRunning(on) {
  document.getElementById("btn-run").disabled = on;
  document.getElementById("btn-stp").disabled = !on;
}

// ── Console helpers ────────────────────────────────────────────────────────
function tagLine(line) {
  const lo = line.toLowerCase();
  if (/error|traceback|exception|✗/.test(lo)) return "err";
  if (/warning|⚠|skipped|skip/.test(lo))      return "warn";
  if (/✓|saved →|complete| ok|done/.test(lo)) return "ok";
  if (/^\s*\[/.test(line))                     return "info";
  if (/^[╔║╚═]/.test(line))                   return "hdr";
  return "ok";
}

function logLine(text, cls) {
  const con  = document.getElementById("console");
  const span = document.createElement("span");
  span.className   = "c-" + (cls || "ok");
  span.textContent = text;
  con.appendChild(span);
  con.scrollTop = con.scrollHeight;
}

function clearConsole() {
  document.getElementById("console").innerHTML = "";
}

// ── Exit ───────────────────────────────────────────────────────────────────
function exitApp() {
  if (currentRunId) stopScript();
  logLine("\n  Shutting down server…\n", "warn");
  fetch("/api/exit", { method: "POST" })
    .catch(() => {})
    .finally(() => {
      setTimeout(() => window.close(), 400);
    });
}
</script>
</body>
</html>
"""

# ── Entry point ────────────────────────────────────────────────────────────────

def _refresh_yf_cookies() -> None:
    """Delete stale yfinance cookie cache and reset the in-memory singleton.

    Yahoo Finance rotates its crumb token periodically.  yfinance persists the
    cookie to ~/Library/Caches/py-yfinance/cookies.db; when that cookie
    expires on Yahoo's side the next API call returns HTTP 401 "Invalid Crumb".

    Deleting cookies.db before the first subprocess launch forces yfinance (in
    every child process) to fetch a fresh cookie/crumb from Yahoo.  We also
    reset the in-memory YfData singleton in case this process itself calls
    yfinance (e.g. the backtest price-history fallback in growthScreener).
    """
    import glob
    cache_dir = os.path.expanduser("~/Library/Caches/py-yfinance")
    removed = []
    for f in glob.glob(os.path.join(cache_dir, "cookies.db*")):
        try:
            os.remove(f)
            removed.append(os.path.basename(f))
        except OSError:
            pass
    if removed:
        print(f"  [yfinance] cleared stale cookie cache: {', '.join(removed)}")

    # Also reset the in-memory singleton so this process re-fetches immediately
    try:
        from yfinance.data import YfData
        yd = YfData()
        yd._crumb  = None
        yd._cookie = None
    except Exception:
        pass   # yfinance not importable yet — subprocesses will handle it


def _port_open(port: int) -> bool:
    """Return True if something is already listening on *port*."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _our_server_alive(url: str) -> bool:
    """Return True if *our* Flask server is responding at *url*."""
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/scripts", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _kill_port(port: int) -> None:
    """Terminate whatever process is bound to *port* and wait until it is free.

    Strategy:
      1. Send SIGTERM (clean shutdown) to each PID holding the port.
      2. Poll up to 4 s for the port to become free.
      3. If still busy, escalate to SIGKILL and poll another 2 s.
    """
    import subprocess, time, socket as _sock
    try:
        def _pids():
            r = subprocess.run(["lsof", "-ti", f":{port}"],
                               capture_output=True, text=True)
            return [p.strip() for p in r.stdout.strip().splitlines()
                    if p.strip().isdigit()]

        def _free():
            with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                s.settimeout(0.1)
                return s.connect_ex(("127.0.0.1", port)) != 0

        # Step 1 — SIGTERM
        for pid in _pids():
            subprocess.run(["kill", pid], capture_output=True)

        # Step 2 — poll up to 4 s
        for _ in range(20):
            time.sleep(0.2)
            if _free():
                return

        # Step 3 — escalate to SIGKILL
        for pid in _pids():
            subprocess.run(["kill", "-9", pid], capture_output=True)

        # Final wait
        for _ in range(10):
            time.sleep(0.2)
            if _free():
                return

    except Exception:
        pass


if __name__ == "__main__":
    import signal as _signal, socket as _socket, time as _time

    parser = argparse.ArgumentParser(description="ValuationSuite web launcher")
    parser.add_argument("--port",       type=int,  default=5050, help="Port (default 5050)")
    parser.add_argument("--no-browser", action="store_true",     help="Don't auto-open browser")
    args = parser.parse_args()

    port = args.port
    url  = f"http://127.0.0.1:{port}"
    _SERVER_PORT = port  # propagated to child processes via VALUATION_SUITE_PORT

    # ── Already running? Just reopen the browser ─────────────────────────────
    if _port_open(port) and _our_server_alive(url):
        print(f"\n  ValuationSuite already running → {url}")
        print("  Reopening browser…\n")
        webbrowser.open(url)
        sys.exit(0)

    # ── Port busy with a dead/foreign process → clear it ─────────────────────
    if _port_open(port):
        print(f"\n  Port {port} is busy — clearing stale process…")
        _kill_port(port)

    # ── Find first free port (try up to +2 in case primary is slow to release) ─
    chosen = None
    for candidate in [port, port + 1, port + 2]:
        if not _port_open(candidate):
            chosen = candidate
            break
    if chosen is None:
        print(f"\n  ERROR: Ports {port}–{port + 2} are all in use.\n"
              "  Close other applications on those ports and try again.")
        sys.exit(1)
    port = chosen
    url  = f"http://127.0.0.1:{port}"
    _SERVER_PORT = port

    # ── Clean SIGTERM handler (macOS Dock → Quit sends SIGTERM) ──────────────
    def _on_sigterm(sig, frame):
        print("\n  ValuationSuite — received stop signal, shutting down…")
        os._exit(0)
    _signal.signal(_signal.SIGTERM, _on_sigterm)

    # ── Clear stale yfinance cookie cache (prevents 401 Invalid Crumb) ───────
    _refresh_yf_cookies()

    # ── Open browser after server is ready ───────────────────────────────────
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=[url]).start()

    print(f"\n  ValuationSuite  →  {url}\n  Ctrl-C to quit\n")
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except OSError as exc:
        print(f"\n  ERROR: Could not bind to port {port}: {exc}")
        sys.exit(1)
