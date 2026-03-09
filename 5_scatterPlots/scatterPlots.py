#!/usr/bin/env python3
"""
scatterPlots.py — Interactive valuation scatter plots for a stock index or custom CSV.

Usage:
    python scatterPlots.py --index SPX [--top 150]
    python scatterPlots.py --index NDX [--top 100]
    python scatterPlots.py --csv /path/to/tickers.csv [--top 200]

Output:
    scatterData/YYYY_MM_DD_scatter_<INDEX>.html
    (opened automatically in the browser)

Six charts, each in its own tab:
    1. Rule of 40 vs EV/S
    2. Revenue Growth vs EV/S
    3. EPS Growth vs P/E
    4. FCF Margin vs EV/FCF
    5. ROE (ROIC proxy) vs EV/EBITDA
    6. Gross Margin vs EV/S
"""

import sys
import os
import argparse
import csv
import datetime
import logging
import random
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from tradingview_screener import Query, col

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    print("\n  ERROR: plotly is not installed.")
    print("  Run: pip install plotly>=5.18.0")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# yfinance emits HTTP 500 / rate-limit errors via its own logger and via
# urllib3 — suppress them so they don't pollute the ValuationSuite console.
for _lib in ("yfinance", "yfinance.base", "yfinance.utils",
             "urllib3", "urllib3.connectionpool",
             "peewee", "charset_normalizer"):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(ROOT, "scatterData")
MAX_WORKERS = 5   # keep well below Yahoo Finance's rate limit

# Colours: teal = below trend (cheap), coral = above trend (expensive)
COL_CHEAP     = "#00c8a0"
COL_EXPENSIVE = "#e05c5c"
COL_TREND     = "#888888"
BG_DARK       = "#141920"
GRID_COL      = "#1e2730"
TEXT_COL      = "#d4d4d4"

INDEX_MAP = {
    "SPX": "S&P 500",
    "NDX": "Nasdaq 100",
    "RUT": "Russell 2000",
    "TSX": "TSX Composite",
}

# Hardcoded constituent lists (same source as valueScreener.py).
# RUT is too large and changes frequently — fetched via exchange fallback.
_SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG",
    "AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BK","BBWI","BAX","BDX","BBY","TECH","BIO","BIIB","BLK","BX","BA","BKNG","BWA","BXP","BSX",
    "BMY","AVGO","BR","BRO","BF.B","BLDR","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL",
    "CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL",
    "CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA",
    "CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DECK","DE","DELL","DAL","DVN","DXCM",
    "FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY",
    "ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS",
    "EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN",
    "IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA",
    "DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
    "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS",
    "LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
    "MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU",
    "MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ",
    "NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
    "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK",
    "SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP",
    "TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB",
    "TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR",
    "VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT",
    "DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WHR","WTW","WRK","WYNN","XEL","XYL",
    "YUM","ZBRA","ZBH","ZTS",
]

_NDX100 = [
    "ADBE","AMD","ABNB","GOOGL","GOOG","AMZN","AEP","AMGN","ADI","ANSS","AAPL","AMAT","APP",
    "ASML","AZN","TEAM","ADSK","ADP","AXON","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CTAS",
    "CSCO","CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX","DXCM","FANG","DDOG",
    "DLTR","EA","EXC","FAST","FTNT","GEHC","GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG",
    "KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI","META","MCHP","MU","MSFT","MRNA",
    "MDLZ","MDB","MNST","NFLX","NVDA","NXPI","ORLY","ON","PCAR","PANW","PAYX","PYPL","PDD","QCOM",
    "REGN","ROP","ROST","SBUX","SNPS","TTWO","TMUS","TSLA","TXN","TTD","VRSK","VRTX","WBD","WBA",
    "WDAY","XEL","ZS","ZM",
]

_TSX = [
    "SHOP","RY","TD","ENB","CP","CNR","BN","BAM","BCE","BMO","BNS","MFC","SLF","TRI","ABX",
    "CCO","IMO","CNQ","SU","CVE","PPL","TRP","K","POW","GWO","IAG","FFH","L","EMA","FTS",
    "H","CAR","REI","CHP","HR","SRU","AP","DIR","NWH","CSH","MRG","CRT","WPM","AEM","AGI",
    "KL","FNV","OR","EDV","MAG","ELD","SSL","IMG","PVG","OGC","TMX","X","ACO","MG","MDA",
    "OTEX","DSG","GIB","BB","CIGI","PHO","ATD","DOL","CTC","MRU","EMP","SAP","QSR","MTY",
    "TFII","TIH","WSP","STN","ATA","BYD","NFI","TDG","RBA","LIF","CCL","ITP","TCL","TXF",
    "WCN","BIN","GFL","SNC","STLC","HPS","FTT","IFP","PRE","FSV","TIXT","AC","CAE","CCA",
    "CHR","WJA","G","YRI","NGD","CG","LUG","BTG","SGY","TVE","BTE","ARX","BIR","CR",
    "ERF","GTE","MEG","NVA","OBE","PEY","PXT","RRX","SCL","SES","SPB","TOU","VET","VLE","WCP",
]

_INDEX_TICKERS = {"SPX": _SP500, "NDX": _NDX100, "TSX": _TSX}

# ── Ticker sourcing ───────────────────────────────────────────────────────────

def _fetch_index_tickers(index: str, top: int) -> list[str]:
    label = INDEX_MAP[index]
    print(f"  Fetching {label} constituents…")

    known = _INDEX_TICKERS.get(index)

    # SPX, NDX, TSX: filter by known constituent list
    if known:
        try:
            _, df = (
                Query()
                .select("name", "market_cap_basic")
                .where(col("name").isin(known), col("is_primary") == True)
                .order_by("market_cap_basic", ascending=False)
                .limit(len(known) + 20)
                .get_scanner_data()
            )
            tickers = df["name"].dropna().tolist()[:top]
            print(f"  {len(tickers)} tickers retrieved")
            return tickers
        except Exception as exc:
            print(f"  WARNING: isin filter failed ({exc}) — trying exchange fallback")

    # RUT (and fallback): fetch NYSE + NASDAQ, sort by market cap, skip top ~503
    # (the S&P 500 large caps) to approximate the Russell 2000 small-cap universe.
    frames = []
    for exch in ("NYSE", "NASDAQ"):
        try:
            _, df = (
                Query()
                .select("name", "market_cap_basic")
                .where(col("exchange") == exch, col("is_primary") == True,
                       col("typespecs").has_none_of("preferred"))
                .order_by("market_cap_basic", ascending=False)
                .limit(1500)
                .get_scanner_data()
            )
            frames.append(df)
        except Exception as exc:
            print(f"  WARNING: {exch} query failed ({exc})")

    if not frames:
        print("  WARNING: all queries failed — returning empty list")
        return []

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["name"]).sort_values(
        "market_cap_basic", ascending=False
    ).reset_index(drop=True)

    if index == "RUT":
        combined = combined.iloc[503:503 + top].reset_index(drop=True)
    else:
        combined = combined.head(top)

    tickers = combined["name"].dropna().tolist()
    print(f"  {len(tickers)} tickers retrieved")
    return tickers


def _tickers_from_str(tickers_str: str, top: int) -> list[str]:
    """Parse a comma-separated ticker string, deduplicate, and return up to `top` symbols."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))   # deduplicate, preserve order
    print(f"  {len(tickers)} tickers parsed from input")
    return tickers[:top]


def _read_csv_tickers(path: str, top: int) -> list[str]:
    tickers = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        # Accept 'ticker', 'symbol', 'Ticker', 'Symbol', or fall back to first column
        fieldnames = reader.fieldnames or []
        col_name = next(
            (f for f in fieldnames if f.lower() in ("ticker", "symbol")),
            fieldnames[0] if fieldnames else None,
        )
        for row in reader:
            if col_name and row.get(col_name):
                tickers.append(row[col_name].strip().upper())
    tickers = [t for t in tickers if t]
    print(f"  {len(tickers)} tickers read from CSV")
    return tickers[:top]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_one(ticker: str, retries: int = 3) -> tuple[str, dict]:
    """Return (ticker, info_dict). Retries up to `retries` times with
    exponential back-off to handle Yahoo Finance rate limits gracefully.
    A small random jitter is added before each attempt to avoid request bursts."""
    for attempt in range(retries):
        time.sleep(random.uniform(0.05, 0.20))   # spread concurrent requests
        try:
            info = yf.Ticker(ticker).info
            if info:
                return ticker, info
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(0.5 * (2 ** attempt))     # 0.5 s, 1.0 s back-off
    return ticker, {}


def _fetch_all(tickers: list[str]) -> dict[str, dict]:
    results = {}
    total   = len(tickers)
    done    = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            tkr, info = fut.result()
            results[tkr] = info
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  Fetched {done}/{total}", end="\r", flush=True)
    print()
    return results


# ── Metric extraction ─────────────────────────────────────────────────────────

def _pct(val) -> float | None:
    """Convert a decimal fraction to percentage, return None if invalid."""
    try:
        v = float(val)
        return v * 100.0 if abs(v) <= 50 else None   # guard against already-pct values
    except (TypeError, ValueError):
        return None


def _num(val) -> float | None:
    try:
        v = float(val)
        return v if v == v else None   # NaN check
    except (TypeError, ValueError):
        return None


def _extract(ticker: str, info: dict) -> dict:
    name         = info.get("longName") or info.get("shortName") or ticker

    rev_growth   = _pct(info.get("revenueGrowth"))
    ebitda_margin= _pct(info.get("ebitdaMargins"))
    rule40       = (rev_growth + ebitda_margin) if (rev_growth is not None and ebitda_margin is not None) else None

    ev_s         = _num(info.get("enterpriseToRevenue"))

    eps_growth   = _pct(info.get("earningsGrowth")) or _pct(info.get("earningsQuarterlyGrowth"))
    pe           = _num(info.get("trailingPE"))

    fcf_raw      = _num(info.get("freeCashflow"))
    rev_raw      = _num(info.get("totalRevenue"))
    ev_raw       = _num(info.get("enterpriseValue"))
    fcf_margin   = (fcf_raw / rev_raw * 100.0) if (fcf_raw and rev_raw) else None
    ev_fcf       = (ev_raw / fcf_raw)           if (fcf_raw and fcf_raw > 0 and ev_raw) else None

    ev_ebitda    = _num(info.get("enterpriseToEbitda"))
    roe          = _pct(info.get("returnOnEquity"))

    gross_margin = _pct(info.get("grossMargins"))

    pb           = _num(info.get("priceToBook"))
    return_52w   = _pct(info.get("fiftyTwoWeekChange"))

    # ── Forward metrics ───────────────────────────────────────────────────────
    # forwardPE and forwardEps are analyst-consensus fields already in .info.
    fwd_pe        = _num(info.get("forwardPE"))
    fwd_eps       = _num(info.get("forwardEps"))
    trail_eps     = _num(info.get("trailingEps"))
    if fwd_eps is not None and trail_eps is not None and trail_eps != 0:
        fwd_eps_growth = (fwd_eps - trail_eps) / abs(trail_eps) * 100.0
    else:
        fwd_eps_growth = None

    # Estimated NTM EV/S: project TTM revenue one year forward at the TTM
    # growth rate.  Not analyst consensus — treat as directional only.
    if ev_raw and rev_raw and rev_raw > 0 and rev_growth is not None:
        proj_rev = rev_raw * (1.0 + rev_growth / 100.0)
        fwd_ev_s = (ev_raw / proj_rev) if proj_rev > 0 else None
    else:
        fwd_ev_s = None

    return {
        "ticker":        ticker,
        "name":          name,
        "rule40":        rule40,
        "ev_s":          ev_s,
        "rev_growth":    rev_growth,
        "eps_growth":    eps_growth,
        "pe":            pe,
        "fcf_margin":    fcf_margin,
        "ev_fcf":        ev_fcf,
        "ev_ebitda":     ev_ebitda,
        "roe":           roe,
        "gross_margin":  gross_margin,
        "pb":            pb,
        "return_52w":    return_52w,
        "fwd_pe":        fwd_pe,
        "fwd_eps_growth": fwd_eps_growth,
        "fwd_ev_s":      fwd_ev_s,
    }


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _trend_line(xs: list, ys: list) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """OLS linear trend line. Returns (x_range, y_range) or (None, None)."""
    if len(xs) < 5:
        return None, None
    try:
        coeffs = np.polyfit(xs, ys, 1)
        x_min, x_max = min(xs), max(xs)
        x_range = np.linspace(x_min, x_max, 200)
        y_range = np.polyval(coeffs, x_range)
        return x_range, y_range
    except Exception:
        return None, None


def _clip_outliers(values: list[float], pct: float = 99.0) -> float:
    """Return the given percentile as a Y-axis ceiling."""
    if not values:
        return None
    return float(np.percentile(values, pct))


def _above_trend(x: float, y: float, coeffs) -> bool:
    return y > np.polyval(coeffs, x)


def _make_scatter(
    rows: list[dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    x_fmt: str = ".1f",
    y_fmt: str = ".2f",
    y_clip_pct: float = 99.0,
    y_floor: float | None = None,
) -> go.Figure:
    """Build a single Plotly scatter figure with trend line and colour coding."""

    # Filter rows to those with both values present and sensible
    valid = [
        r for r in rows
        if r.get(x_key) is not None and r.get(y_key) is not None
        and -500 < r[x_key] < 2000    # basic sanity guard
        and 0 < r[y_key] < 1e6
    ]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=TEXT_COL))
        _style_fig(fig, title, x_label, y_label)
        return fig

    xs = [r[x_key] for r in valid]
    ys = [r[y_key] for r in valid]

    # Clip extreme Y values for display
    y_max = _clip_outliers(ys, y_clip_pct)

    # ── OLS subset: exclude extreme-X outliers robustly on any dataset size ──
    # Sort by X and drop the outermost `n_excl` points from each end.
    # This handles small CSVs (15 tickers) where a 1st-percentile cut is useless,
    # as well as large indices where a handful of extreme tickers skew the slope.
    n_excl = max(1, int(len(xs) * 0.05))          # ~5 % from each end, min 1
    n_excl = min(n_excl, max(1, len(xs) // 5))    # never more than 20 % from each end
    _sorted = sorted(zip(xs, ys), key=lambda t: t[0])
    inner   = _sorted[n_excl : len(_sorted) - n_excl] if len(_sorted) > 2 * n_excl else _sorted
    _y_lo   = y_floor if y_floor is not None else 0.0
    _y_hi   = y_max   if y_max  is not None else float(np.percentile(ys, 99))
    ols_pts = [(x, y) for x, y in inner if _y_lo <= y <= _y_hi]
    _ols_ok = len(ols_pts) >= 5
    xs_ols  = [x for x, y in ols_pts]
    ys_ols  = [y for x, y in ols_pts]

    # Compute trend coefficients on the inner subset
    try:
        coeffs = np.polyfit(xs_ols if _ols_ok else xs,
                            ys_ols if _ols_ok else ys, 1)
    except Exception:
        coeffs = None

    # Colour each point
    colours = []
    for r in valid:
        if coeffs is not None:
            colours.append(COL_EXPENSIVE if _above_trend(r[x_key], r[y_key], coeffs) else COL_CHEAP)
        else:
            colours.append(COL_CHEAP)

    # Hover text
    hover = [
        f"<b>{r['ticker']}</b>  {r['name']}<br>"
        f"{x_label}: {r[x_key]:{x_fmt}}<br>"
        f"{y_label}: {r[y_key]:{y_fmt}}"
        for r in valid
    ]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=[r["ticker"] for r in valid],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_COL),
        hovertext=hover,
        hoverinfo="text",
        marker=dict(color=colours, size=8, opacity=0.85,
                    line=dict(width=0.5, color="rgba(0,0,0,0.25)")),
        showlegend=False,
    ))

    # Trend line — span the OLS inner range only (avoids extending into extreme
    # outlier territory), then TRIM (not clamp) to [y_floor, y_max] so there is
    # never a flat horizontal segment at the floor.
    if coeffs is not None and _ols_ok:
        x_tl = np.linspace(min(xs_ols), max(xs_ols), 200)
        y_tl = np.polyval(coeffs, x_tl)
        mask = np.ones(len(x_tl), dtype=bool)
        if y_floor is not None:
            mask &= (y_tl >= y_floor)
        if y_max is not None:
            mask &= (y_tl <= y_max)
        if mask.sum() >= 2:
            fig.add_trace(go.Scatter(
                x=x_tl[mask].tolist(), y=y_tl[mask].tolist(),
                mode="lines",
                line=dict(color=COL_TREND, width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Legend annotations (colour key)
    fig.add_annotation(
        text="● Expensive vs peers", xref="paper", yref="paper",
        x=0.99, y=0.99, showarrow=False,
        font=dict(size=11, color=COL_EXPENSIVE), xanchor="right"
    )
    fig.add_annotation(
        text="● Cheap vs peers", xref="paper", yref="paper",
        x=0.99, y=0.955, showarrow=False,
        font=dict(size=11, color=COL_CHEAP), xanchor="right"
    )

    _style_fig(fig, title, x_label, y_label, y_max=y_max, y_floor=y_floor)
    return fig


def _style_fig(fig: go.Figure, title: str, x_label: str, y_label: str,
               y_max: float | None = None, y_floor: float | None = None) -> None:
    y_range    = [y_floor if y_floor is not None else None, y_max]
    _fix_range = y_floor is not None or y_max is not None
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT_COL), x=0.5, xanchor="center"),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color=TEXT_COL, family="Menlo, Monaco, 'Courier New', monospace"),
        xaxis=dict(
            title=x_label,
            gridcolor=GRID_COL, zerolinecolor=GRID_COL,
            showspikes=True, spikecolor="#555", spikethickness=1,
        ),
        yaxis=dict(
            title=y_label,
            gridcolor=GRID_COL, zerolinecolor=GRID_COL,
            range=y_range,
            autorange=False if _fix_range else True,
            showspikes=True, spikecolor="#555", spikethickness=1,
        ),
        hoverlabel=dict(bgcolor="#1e2730", bordercolor="#333", font_size=12),
        autosize=True,
        margin=dict(l=60, r=20, t=60, b=60),
        dragmode="pan",
    )


# ── Chart descriptions ────────────────────────────────────────────────────────

CHART_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "Rule of 40": (
        "What it shows",
        "The Rule of 40 adds a company's revenue growth rate and EBITDA margin. A score above 40 is "
        "considered healthy — it means the business is balancing growth and profitability well. "
        "This chart plots that efficiency score against the valuation multiple (EV/Revenue) the market "
        "assigns to it. The dashed trend line represents the 'fair' multiple for a given Rule of 40 score "
        "across the index. <b>Teal points</b> trade below the trend — they look cheap relative to their "
        "efficiency. <b>Coral points</b> carry a premium. The most interesting opportunities are typically "
        "teal companies in the upper-right (high Rule of 40, still below trend) and coral companies in "
        "the upper-left (low Rule of 40, but expensive — priced for perfection they haven't delivered).",
    ),
    "Rev Growth": (
        "What it shows",
        "The simplest version of growth vs. valuation. Revenue growth is plotted on the X-axis; "
        "EV/Revenue on the Y-axis. The trend line captures the market's average willingness to pay "
        "per unit of growth across the index. <b>Teal points below the line</b> are cheap growers — "
        "the market hasn't fully priced their growth rate relative to peers. <b>Coral points above "
        "the line</b> are expensive relative to their growth. The ideal candidates sit in the lower-right "
        "corner: high revenue growth, low multiple. Slow growers with high multiples (upper-left) "
        "are the most vulnerable to re-rating.",
    ),
    "PEG (EPS Growth)": (
        "What it shows",
        "This is the PEG ratio visualised as a scatter plot. A PEG ratio of 1× means the P/E equals "
        "the earnings growth rate — historically considered fair value. Points below the trend line have "
        "a low implied PEG (cheap earnings growth); points above have a high implied PEG (expensive). "
        "Note that P/E is less meaningful for unprofitable companies, and EPS growth can be distorted "
        "by one-time items or tax changes, so use this alongside the FCF and EV/S charts rather than "
        "in isolation. Companies with negative EPS growth or negative P/E are excluded.",
    ),
    "FCF Quality": (
        "What it shows",
        "Free cash flow is harder to manipulate than EBITDA — it requires real cash to arrive in the "
        "bank after capex. FCF margin (FCF as a percentage of revenue) measures how efficiently a "
        "company converts sales into cash. EV/FCF is the valuation multiple paid for that cash. "
        "<b>Teal points</b> generate strong free cash flow relative to their price. "
        "<b>Coral points</b> carry a premium that may not be justified by current FCF. "
        "Asset-heavy businesses (utilities, manufacturing) naturally show lower FCF margins than "
        "software or financial companies, so this chart is most useful when comparing within a sector.",
    ),
    "ROIC / EV·EBITDA": (
        "What it shows",
        "Return on equity (used here as a ROIC proxy) measures how effectively management deploys "
        "capital. High ROE sustained over time is a hallmark of compounding businesses with durable "
        "competitive advantages. EV/EBITDA is the valuation multiple paid for that capital efficiency. "
        "<b>Teal points</b> are priced cheaply relative to their return profile — the market may be "
        "underappreciating their capital efficiency. <b>Coral points</b> in the upper-left (high "
        "multiple, low ROE) are pricing in future improvement that hasn't yet materialised. "
        "Note: ROE can be inflated by financial leverage; treat high-ROE readings in capital-light "
        "businesses (tech, consumer brands) differently from those in leveraged financials.",
    ),
    "Gross Margin": (
        "What it shows",
        "Gross margin reflects pricing power and the scalability of the underlying business model — "
        "it shows how much revenue remains after direct costs, before SG&A and R&D are deducted. "
        "This is particularly useful for early-stage or high-growth companies where EBITDA is still "
        "negative: a high gross margin signals that profitability is a deliberate investment choice, "
        "not a structural problem. Software and SaaS companies typically cluster in the upper-right "
        "(70–85% gross margin, high EV/S). <b>Teal points</b> carry a lower multiple than their gross "
        "margin quality suggests. <b>Coral points</b> are priced at a premium relative to margin peers.",
    ),
    "P/B vs ROE": (
        "What it shows",
        "This is the Buffett / Graham valuation framework in scatter form. Price-to-book measures how "
        "much investors pay for each dollar of net assets; return on equity measures how productively "
        "management deploys those assets. A company that consistently earns high ROE deserves a high "
        "P/B — the trend line captures that relationship for the current index. "
        "<b>Teal points below the line</b> are the most interesting: high ROE but lower P/B than peers "
        "with similar returns — the market is underpricing their capital efficiency. "
        "<b>Coral points above the line</b> trade at a premium relative to their ROE — either the "
        "market is pricing in future improvement, or the stock is simply expensive. "
        "Note: ROE can be inflated by share buybacks or leverage; cross-reference with the FCF and "
        "ROIC charts for confirmation.",
    ),
    "52W Return vs EV/S": (
        "What it shows",
        "This chart overlays price momentum with valuation to identify re-rating candidates and "
        "stretched trades. The X-axis shows the stock's total return over the past 52 weeks; "
        "the Y-axis is EV/Revenue. The trend line captures the index's average willingness to "
        "pay a higher multiple for stocks that have already moved. "
        "<b>Teal points</b> — strong 12-month performance but still trading below the valuation "
        "implied by that momentum — suggest a re-rating may be underway and the market hasn't "
        "fully caught up. These are often the most actionable setups. "
        "<b>Coral points</b> in the upper-right have already moved and already carry a premium — "
        "the easy money may be made. Stocks in the lower-left (negative return, cheap multiple) "
        "are either value traps or early turnarounds — worth investigating but not buying blindly.",
    ),
    "Rev Growth vs Gross Margin": (
        "What it shows",
        "A pure business-quality chart with no valuation axis — this is about identifying which "
        "companies are building durable competitive advantages. Revenue growth shows momentum; "
        "gross margin shows pricing power and the scalability of the cost structure. "
        "The upper-right quadrant (high growth + high gross margin) is the 'magic zone' — "
        "typically premium SaaS, platforms, and asset-light compounders. These businesses can "
        "afford to invest heavily in growth because the unit economics are structurally sound. "
        "The lower-left (low growth + low margin) represents commoditised or structurally challenged "
        "businesses. The trend line shows the index's average gross margin for a given growth rate. "
        "<b>Teal points</b> have higher margins than typical for their growth rate — better business "
        "quality than peers growing at the same speed. <b>Coral points</b> have lower margins than "
        "expected — their growth may be more expensive to sustain.",
    ),
    # ── Forward chart descriptions ─────────────────────────────────────────────
    "Fwd Rule of 40": (
        "What it shows",
        "The TTM Rule of 40 score (revenue growth + EBITDA margin) plotted against analyst-consensus "
        "Forward P/E. This asks: 'for a given level of combined growth and profitability, how many "
        "times next year's earnings is the market willing to pay?' <b>Teal points</b> run a strong "
        "Rule of 40 but carry a modest forward earnings multiple — the market may not yet be pricing "
        "in the operational efficiency. <b>Coral points</b> carry a premium forward multiple relative "
        "to their efficiency score. Unlike plotting Rule of 40 against forward EV/S, using Forward P/E "
        "avoids a mechanical inverse correlation caused by high growth deflating the estimated forward "
        "revenue denominator.",
    ),
    "Fwd Rev Growth": (
        "What it shows",
        "TTM revenue growth against analyst-consensus Forward P/E. Useful for identifying where the "
        "market is paying a high earnings multiple for companies that haven't yet shown strong top-line "
        "growth, and vice versa. <b>Teal points</b> with strong TTM revenue growth and a low forward "
        "P/E are priced conservatively — the market may be sceptical the growth will persist, creating "
        "a potential opportunity if it does. <b>Coral points</b> with modest growth and a high forward "
        "P/E are pricing in a future improvement that hasn't materialised yet.",
    ),
    "Fwd PEG": (
        "What it shows",
        "The forward PEG ratio visualised as a scatter — using analyst-consensus forward P/E "
        "on the Y-axis and forward EPS growth (derived from consensus forward EPS vs trailing EPS) "
        "on the X-axis. Both inputs are consensus-based, making this more reliable than the TTM "
        "PEG chart. A forward PEG near 1× (P/E equals growth rate) is historically considered "
        "fair value. <b>Teal points</b> have a low implied forward PEG — the market is pricing "
        "earnings growth cheaply relative to consensus expectations. <b>Coral points</b> carry "
        "a premium; the market expects the growth to be sustained or accelerate further.",
    ),
    "Fwd FCF": (
        "What it shows",
        "TTM FCF margin against estimated forward EV/Revenue. Companies with strong TTM free "
        "cash flow and a low forward multiple are particularly interesting — the market may be "
        "applying a 'revenue slowdown' discount that undervalues the underlying cash generation "
        "quality. Since FCF is hard to inflate and changes slowly, a high TTM FCF margin paired "
        "with a cheap forward multiple is one of the more durable valuation signals. "
        "<b>Teal points</b> fit this profile; <b>coral points</b> carry a full forward premium "
        "for their cash conversion.",
    ),
    "Fwd Gross Margin": (
        "What it shows",
        "TTM gross margin (a slow-moving structural quality indicator) against estimated forward "
        "EV/Revenue. High gross margin reflects pricing power that compounds over time — using it "
        "against a forward multiple helps identify companies where the market is discounting a "
        "durable competitive advantage. <b>Teal points</b> have structural quality priced at a "
        "modest forward multiple. <b>Coral points</b> are paying a full forward premium for their "
        "margin profile — justified if growth is accelerating, risky if it isn't.",
    ),
    "Fwd ROIC": (
        "What it shows",
        "TTM return on equity (capital efficiency proxy) against analyst-consensus forward P/E. "
        "The forward Buffett framework: the market sets a forward earnings multiple that should "
        "reflect how well the business compounds capital going forward. <b>Teal points</b> with "
        "high ROE and low forward P/E are the most compelling — the market is pricing in "
        "mean-reversion in a business that has historically deployed capital well. "
        "<b>Coral points</b> with low ROE and high forward P/E are pricing in a turnaround in "
        "capital efficiency that consensus earnings already embed.",
    ),
    "Fwd Momentum": (
        "What it shows",
        "52-week price return against estimated forward EV/Revenue. The momentum + forward "
        "valuation combination highlights the most actionable re-rating setups. <b>Teal points</b> "
        "with strong momentum but low forward multiples suggest the market hasn't fully priced "
        "the move — potential continued re-rating. <b>Coral points</b> in the upper-right are "
        "the highest-risk setup: expensive on a forward basis and already priced for perfection. "
        "Stocks in the lower-left (negative return, low forward multiple) are either unloved "
        "value opportunities or value traps — the quality tabs help distinguish which.",
    ),
}


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _build_html(
    charts_ttm: list[tuple[str, go.Figure]],
    charts_fwd: list[tuple[str, go.Figure]],
    index_label: str,
    n_tickers: int,
) -> str:
    """
    Assemble a self-contained HTML page with a TTM / Forward mode bar and
    per-mode chart tab bars.  The very first figure exports Plotly.js via CDN.
    """
    today = datetime.date.today().strftime("%B %d, %Y")

    # ── render one mode section ──────────────────────────────────────────────
    def _render_section(charts, prefix, first_chart_idx):
        tab_ids = [f"{prefix}-tab{i}" for i in range(len(charts))]
        divs    = []
        oi      = first_chart_idx
        for i, (_, fig) in enumerate(charts):
            div = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if oi == 0 else False,
                config={"displayModeBar": True, "scrollZoom": True,
                        "modeBarButtonsToRemove": ["toImage"],
                        "responsive": True},
                div_id=f"chart-{tab_ids[i]}",
            )
            divs.append(div)
            oi += 1

        tab_buttons = "\n".join(
            f'  <button class="tab-btn{" active" if i == 0 else ""}" '
            f'onclick="showTab(\'{prefix}\', \'{tab_ids[i]}\')" id="btn-{tab_ids[i]}">'
            f'{label}</button>'
            for i, (label, _) in enumerate(charts)
        )

        pane_parts = []
        for i, (label, _) in enumerate(charts):
            desc = CHART_DESCRIPTIONS.get(label, ("", ""))
            desc_html = (
                f'<div class="chart-desc">'
                f'<span class="desc-heading">{desc[0]} &mdash;</span> '
                f'{desc[1]}'
                f'</div>'
            ) if desc[1] else ""
            pane_parts.append(
                f'<div class="chart-pane{" active" if i == 0 else ""}" id="pane-{tab_ids[i]}">'
                f'\n{divs[i]}\n'
                f'{desc_html}'
                f'</div>'
            )
        return tab_buttons, "\n".join(pane_parts), oi

    ttm_tabs, ttm_panes, ni = _render_section(charts_ttm, "ttm", 0)
    fwd_tabs, fwd_panes, _  = _render_section(charts_fwd,  "fwd", ni)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scatter Plots — {index_label}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body  {{ background: {BG_DARK}; color: {TEXT_COL};
           font-family: Menlo, Monaco, "Courier New", monospace; }}
  header {{ display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 10px;
            padding: 18px 28px 10px;
            border-bottom: 1px solid #1e2730; }}
  header .meta h1 {{ font-size: 1.25rem; font-weight: 600; color: #e8e8e8; }}
  header .meta p  {{ font-size: 0.78rem; color: #888; margin-top: 4px; }}
  .search-wrap  {{ display: flex; align-items: center; gap: 8px;
                   background: #1a2330; border: 1px solid #2a3540;
                   border-radius: 6px; padding: 7px 12px; flex-shrink: 0; }}
  .search-lbl   {{ font-size: 0.72rem; color: #666; white-space: nowrap; }}
  #ticker-search {{ background: transparent; border: none; outline: none;
                    color: #e8e8e8; font-family: inherit; font-size: 0.82rem;
                    width: 90px; letter-spacing: 0.05em; }}
  #ticker-search::placeholder {{ color: #444; }}
  #search-status {{ font-size: 0.72rem; min-width: 60px; }}
  /* ── mode bar ────────────────────────────────────────────────────────── */
  .mode-bar {{ display: flex; gap: 8px; padding: 12px 20px 10px;
               border-bottom: 1px solid #1e2730; }}
  .mode-btn {{ background: #1a2330; border: 1px solid #2a3540; color: #aaa;
               padding: 7px 22px; border-radius: 6px; cursor: pointer;
               font-size: 0.82rem; font-family: inherit; font-weight: 600;
               transition: background 0.15s, color 0.15s; letter-spacing: 0.04em; }}
  .mode-btn:hover  {{ background: #243040; color: #ccc; }}
  .mode-btn.active {{ background: {COL_CHEAP}; color: #0a1015; border-color: {COL_CHEAP}; }}
  .mode-section        {{ display: none; }}
  .mode-section.active {{ display: block; }}
  /* ── chart tabs ──────────────────────────────────────────────────────── */
  .tab-bar  {{ display: flex; gap: 4px; padding: 14px 20px 0;
               border-bottom: 1px solid #1e2730; flex-wrap: wrap; }}
  .tab-btn  {{ background: #1a2330; border: 1px solid #2a3540;
               color: #aaa; padding: 7px 18px; border-radius: 6px 6px 0 0;
               cursor: pointer; font-size: 0.78rem; font-family: inherit;
               transition: background 0.15s, color 0.15s; white-space: nowrap; }}
  .tab-btn:hover  {{ background: #243040; color: #ccc; }}
  .tab-btn.active {{ background: {BG_DARK}; color: #e8e8e8;
                     border-bottom-color: {BG_DARK}; }}
  .chart-pane        {{ display: none; padding: 10px 20px 0; }}
  .chart-pane.active {{ display: block; }}
  .chart-pane > div  {{ width: 100%; height: calc(100vh - 190px); }}
  .chart-desc  {{ margin: 18px 4px 28px; padding: 16px 20px;
                  background: #1a2330; border-left: 3px solid #2a4060;
                  border-radius: 0 6px 6px 0;
                  font-size: 0.8rem; line-height: 1.65; color: #a0aab8; }}
  .desc-heading {{ color: #d4d4d4; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <div class="meta">
    <h1>Scatter Plots — {index_label}</h1>
    <p>{n_tickers} tickers &nbsp;·&nbsp; Generated {today} &nbsp;·&nbsp;
       Teal = cheap vs peers &nbsp;·&nbsp; Coral = expensive vs peers &nbsp;·&nbsp;
       Dashed line = OLS trend</p>
  </div>
  <div class="search-wrap">
    <span class="search-lbl">Find ticker</span>
    <input type="text" id="ticker-search" placeholder="e.g. NVDA"
           oninput="onSearch(event)" autocomplete="off" spellcheck="false" maxlength="12">
    <span id="search-status"></span>
  </div>
</header>
<nav class="mode-bar">
  <button class="mode-btn active" onclick="setMode('ttm')" id="modebtn-ttm">TTM</button>
  <button class="mode-btn" onclick="setMode('fwd')" id="modebtn-fwd">Forward (est.)</button>
</nav>
<section id="section-ttm" class="mode-section active">
<nav class="tab-bar">
{ttm_tabs}
</nav>
{ttm_panes}
</section>
<section id="section-fwd" class="mode-section">
<nav class="tab-bar">
{fwd_tabs}
</nav>
{fwd_panes}
</section>
<script>
var _curMode = "ttm";

function setMode(mode) {{
  _curMode = mode;
  document.querySelectorAll(".mode-section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("section-" + mode).classList.add("active");
  document.getElementById("modebtn-" + mode).classList.add("active");
  var el = _activeEl();
  if (el && window.Plotly) window.Plotly.Plots.resize(el);
  _reapplySearch();
}}

function showTab(mode, id) {{
  var section = document.getElementById("section-" + mode);
  section.querySelectorAll(".chart-pane").forEach(p => p.classList.remove("active"));
  section.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("pane-" + id).classList.add("active");
  document.getElementById("btn-"  + id).classList.add("active");
  var divEl = document.querySelector("#pane-" + id + " [id^='chart-']");
  if (divEl && window.Plotly) {{
    window.Plotly.Plots.resize(divEl);
    _reapplySearch();
  }}
}}

// ── Ticker search & highlight ─────────────────────────────────────────────
var _origMkr   = {{}};   // chartId → {{ colors, size }}
var _lastQuery = "";

function _activeEl() {{
  var section = document.querySelector(".mode-section.active");
  if (!section) return null;
  var pane = section.querySelector(".chart-pane.active");
  return pane ? pane.querySelector("[id^='chart-']") : null;
}}

function _cache(el) {{
  if (!el || !el.data || !el.data[0] || _origMkr[el.id]) return;
  var m = el.data[0].marker;
  _origMkr[el.id] = {{
    colors: Array.isArray(m.color) ? m.color.slice() : m.color,
    size:   m.size,
  }};
}}

function _restore(el) {{
  if (!el || !_origMkr[el.id]) return;
  var o = _origMkr[el.id];
  Plotly.restyle(el, {{
    "marker.color":   [o.colors],
    "marker.size":    [o.size],
    "marker.opacity": [0.85],
  }}, [0]);
}}

function _highlight(el, query) {{
  if (!el || !el.data || !el.data[0]) return;
  _cache(el);
  var tickers = el.data[0].text;
  if (!tickers || !tickers.length) return;
  var q = query.toUpperCase();

  // exact match → prefix match → substring match
  var idx = -1;
  for (var i = 0; i < tickers.length; i++) {{
    if (tickers[i].toUpperCase() === q) {{ idx = i; break; }}
  }}
  if (idx === -1) {{
    for (var i = 0; i < tickers.length; i++) {{
      if (tickers[i].toUpperCase().startsWith(q)) {{ idx = i; break; }}
    }}
  }}
  if (idx === -1) {{
    for (var i = 0; i < tickers.length; i++) {{
      if (tickers[i].toUpperCase().includes(q)) {{ idx = i; break; }}
    }}
  }}

  var statusEl = document.getElementById("search-status");
  if (idx === -1) {{
    _restore(el);
    statusEl.textContent = "not found";
    statusEl.style.color = "#e05c5c";
    return;
  }}

  var o      = _origMkr[el.id];
  var n      = tickers.length;
  var oSize  = typeof o.size === "number" ? o.size : 8;
  var oCols  = Array.isArray(o.colors) ? o.colors : Array(n).fill(o.colors);
  var newColors  = oCols.map(function(c, i) {{ return i === idx ? "#FFD700" : c; }});
  var newSizes   = Array.from({{length: n}}, function(_, i) {{ return i === idx ? oSize * 2.5 : oSize; }});
  var newOpacity = Array.from({{length: n}}, function(_, i) {{ return i === idx ? 1.0 : 0.18; }});

  Plotly.restyle(el, {{
    "marker.color":   [newColors],
    "marker.size":    [newSizes],
    "marker.opacity": [newOpacity],
  }}, [0]);

  statusEl.textContent = tickers[idx];
  statusEl.style.color = "#FFD700";
}}

function onSearch(e) {{
  _lastQuery = e.target.value.trim();
  var el       = _activeEl();
  var statusEl = document.getElementById("search-status");
  if (!_lastQuery) {{
    if (el) _restore(el);
    statusEl.textContent = "";
    return;
  }}
  _highlight(el, _lastQuery);
}}

function _reapplySearch() {{
  if (!_lastQuery) return;
  _highlight(_activeEl(), _lastQuery);
}}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Valuation scatter plots for a stock index.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--index", choices=list(INDEX_MAP), default="SPX",
                     help="Index to screen (default: SPX)")
    grp.add_argument("--csv",   metavar="PATH",
                     help="Path to a CSV file with a 'ticker' or 'symbol' column")
    parser.add_argument("--tickers", metavar="T1,T2,...",
                        help="Comma-separated ticker symbols — overrides --index and --csv")
    parser.add_argument("--top", type=int, default=150,
                        help="Maximum number of tickers to analyse (default: 150)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Get tickers ─────────────────────────────────────────────────────────
    if args.tickers:
        tickers     = _tickers_from_str(args.tickers, args.top)
        index_label = "Custom Tickers"
        slug        = "custom"
    elif args.csv:
        tickers     = _read_csv_tickers(args.csv, args.top)
        index_label = f"Custom CSV ({os.path.basename(args.csv)})"
        slug        = "csv"
    else:
        tickers     = _fetch_index_tickers(args.index, args.top)
        index_label = f"{INDEX_MAP[args.index]} ({args.index})"
        slug        = args.index

    if not tickers:
        print("  No tickers found — nothing to plot.")
        sys.exit(1)

    # ── 2. Fetch data ──────────────────────────────────────────────────────────
    print(f"\n  Fetching financials for {len(tickers)} tickers…")
    raw = _fetch_all(tickers)

    rows = [_extract(t, raw[t]) for t in tickers]
    n    = len(rows)
    print(f"  Data ready for {n} tickers\n")

    # ── 3. Build charts ────────────────────────────────────────────────────────
    print("  Building charts…")
    charts = [
        # ── 1. Most universal efficiency screen ───────────────────────────────
        ("Rule of 40", _make_scatter(
            rows, "rule40", "ev_s",
            x_label="Rule of 40 (Rev Growth % + EBITDA Margin %)",
            y_label="EV / Revenue",
            title="EV/S vs Rule of 40",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 2. Buffett / Graham fundamental value framework ───────────────────
        ("P/B vs ROE", _make_scatter(
            rows, "roe", "pb",
            x_label="Return on Equity % (ROE)",
            y_label="Price / Book",
            title="P/B vs ROE  (Buffett framework)",
            x_fmt=".1f", y_fmt=".2f",
            y_floor=0,
        )),
        # ── 3. Core growth-to-valuation ───────────────────────────────────────
        ("Rev Growth", _make_scatter(
            rows, "rev_growth", "ev_s",
            x_label="Revenue Growth YoY (%)",
            y_label="EV / Revenue",
            title="EV/S vs Revenue Growth",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 4. Cash-flow quality (hard to manipulate) ─────────────────────────
        ("FCF Quality", _make_scatter(
            rows, "fcf_margin", "ev_fcf",
            x_label="Free Cash Flow Margin (%)",
            y_label="EV / Free Cash Flow",
            title="EV/FCF vs FCF Margin",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 5. Earnings-based PEG ─────────────────────────────────────────────
        ("PEG (EPS Growth)", _make_scatter(
            rows, "eps_growth", "pe",
            x_label="EPS Growth YoY (%)",
            y_label="Trailing P/E",
            title="P/E vs EPS Growth  (PEG visualised)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 6. Momentum meets valuation ───────────────────────────────────────
        ("52W Return vs EV/S", _make_scatter(
            rows, "return_52w", "ev_s",
            x_label="52-Week Price Return (%)",
            y_label="EV / Revenue",
            title="EV/S vs 52-Week Return",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 7. Capital-efficiency screen ──────────────────────────────────────
        ("ROIC / EV·EBITDA", _make_scatter(
            rows, "roe", "ev_ebitda",
            x_label="Return on Equity % (ROIC proxy)",
            y_label="EV / EBITDA",
            title="EV/EBITDA vs ROE  (ROIC proxy)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 8. Pricing-power / margin quality ────────────────────────────────
        ("Gross Margin", _make_scatter(
            rows, "gross_margin", "ev_s",
            x_label="Gross Margin (%)",
            y_label="EV / Revenue",
            title="EV/S vs Gross Margin",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        # ── 9. Pure business-model quality (no valuation axis) ───────────────
        ("Rev Growth vs Gross Margin", _make_scatter(
            rows, "rev_growth", "gross_margin",
            x_label="Revenue Growth YoY (%)",
            y_label="Gross Margin (%)",
            title="Gross Margin vs Revenue Growth  (quality quadrant)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
    ]

    # ── 4. Build forward charts ────────────────────────────────────────────────
    print("  Building forward charts…")
    charts_fwd = [
        ("Fwd Rule of 40", _make_scatter(
            rows, "rule40", "fwd_pe",
            x_label="Rule of 40 (Rev Growth % + EBITDA Margin %)",
            y_label="Forward P/E (consensus)",
            title="Fwd P/E vs Rule of 40",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd Rev Growth", _make_scatter(
            rows, "rev_growth", "fwd_pe",
            x_label="Revenue Growth YoY (%)",
            y_label="Forward P/E (consensus)",
            title="Fwd P/E vs Revenue Growth",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd PEG", _make_scatter(
            rows, "fwd_eps_growth", "fwd_pe",
            x_label="Fwd EPS Growth (consensus, %)",
            y_label="Forward P/E",
            title="Fwd P/E vs Fwd EPS Growth  (Forward PEG)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd FCF", _make_scatter(
            rows, "fcf_margin", "fwd_ev_s",
            x_label="TTM Free Cash Flow Margin (%)",
            y_label="Est. NTM EV / Revenue",
            title="Fwd EV/S vs FCF Margin  (NTM est.)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd Gross Margin", _make_scatter(
            rows, "gross_margin", "fwd_ev_s",
            x_label="Gross Margin (%)",
            y_label="Est. NTM EV / Revenue",
            title="Fwd EV/S vs Gross Margin  (NTM est.)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd ROIC", _make_scatter(
            rows, "roe", "fwd_pe",
            x_label="Return on Equity % (ROIC proxy)",
            y_label="Forward P/E",
            title="Fwd P/E vs ROE  (ROIC proxy)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
        ("Fwd Momentum", _make_scatter(
            rows, "return_52w", "fwd_ev_s",
            x_label="52-Week Price Return (%)",
            y_label="Est. NTM EV / Revenue",
            title="Fwd EV/S vs 52-Week Return  (NTM est.)",
            x_fmt=".1f", y_fmt=".1f",
            y_floor=0,
        )),
    ]

    # ── 5. Write HTML ──────────────────────────────────────────────────────────
    html    = _build_html(charts, charts_fwd, index_label, n)
    today   = datetime.date.today().strftime("%Y_%m_%d")
    outfile = os.path.join(OUTPUT_DIR, f"{today}_scatter_{slug}.html")

    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_kb = os.path.getsize(outfile) / 1024
    print(f"  Saved: {outfile}  ({size_kb:.0f} KB)")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        print("  Opening in browser…")
        webbrowser.open("file://" + os.path.abspath(outfile))

    print("\nDone.")


if __name__ == "__main__":
    main()
