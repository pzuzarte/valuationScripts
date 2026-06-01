"""
Market Concentration & Macro Valuation Dashboard
==================================================
Pulls together four long-run market-valuation gauges into a single HTML report:

  1. S&P 500 Concentration -- top-10 stocks as % of index market cap.
     Live data from TradingView screener; historical anchor points from
     academic research (Mauboussin, Goldman Sachs) back to 1927.

  2. Buffett Indicator -- Total equity market cap / nominal GDP (%).
     Source: FRED NCBEILQ027S / GDP.  History from 1945.

  3. Shiller CAPE (Cyclically Adjusted P/E) -- 10-year real earnings average.
     Source: Robert Shiller (Yale).  History from 1871.

  4. Fed Model -- S&P 500 earnings yield minus 10-Year Treasury yield.
     Source: Shiller CAPE earnings yield + FRED GS10.  History from 1953.

All charts are embedded in the HTML as base-64 PNG images (no CDN required).

USAGE
-----
  python marketConcentration.py
  python marketConcentration.py --no-concentration   # skip slow TV screener fetch
"""

import sys, os, io, math, datetime, argparse, base64, logging
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Chart colour palette ──────────────────────────────────────────────────────
BG    = "#0f1117"
S1    = "#161a27"
S2    = "#1a1e2e"
BD    = "#252a3a"
TEXT  = "#e2e8f0"
SUB   = "#94a3b8"
LINE1 = "#6366f1"
LINE2 = "#14b8a6"
LINE3 = "#f59e0b"
RED   = "#ef4444"
GRN   = "#22c55e"
YLW   = "#eab308"

# ── Historical concentration anchor points ────────────────────────────────────
# Top-10 stocks as % of S&P 500 (or predecessor US large-cap indices).
# Sources: Mauboussin / Callahan (Credit Suisse 2017),
#          Goldman Sachs Global Investment Research,
#          Howard Silverblatt (S&P Dow Jones Indices).
CONCENTRATION_HIST = [
    (1927, 36), (1932, 42), (1939, 40), (1950, 37),
    (1960, 35), (1965, 38), (1970, 30), (1975, 24),
    (1980, 23), (1985, 20), (1990, 19), (1995, 21),
    (2000, 25), (2005, 20), (2010, 19), (2015, 18),
    (2018, 21), (2020, 27), (2021, 30), (2022, 27),
]

# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concentrationData")

def _fred(series_id, start=None):
    """
    Download a FRED series as a tidy DataFrame (date, value).  No API key.

    Primary: pandas_datareader (uses FRED's JSON API, different code path).
    Fallback: direct CSV download via requests.
    Both paths cache to a local CSV (7-day TTL) so a second run works even
    when FRED is temporarily unreachable.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(_CACHE_DIR, f"fred_{series_id}.csv")
    start_dt   = start or "1945-01-01"

    def _to_tidy(df_raw):
        """Normalise whatever shape comes back to a (date-indexed, 'value') df."""
        if isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index.name = "date"
        else:
            df_raw = df_raw.reset_index()
            df_raw.columns = ["date", "value"]
            df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
            df_raw = df_raw.dropna(subset=["date"]).set_index("date")
        # Rename the series column to 'value'
        if "value" not in df_raw.columns:
            df_raw = df_raw.rename(columns={df_raw.columns[0]: "value"})
        df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
        return df_raw[["value"]].dropna().sort_index()

    def _cache_and_return(df):
        df.reset_index().to_csv(cache_file, index=False)
        return df

    def _load_cache():
        if not os.path.isfile(cache_file):
            return None
        age_days = (datetime.datetime.now().timestamp()
                    - os.path.getmtime(cache_file)) / 86400
        if age_days > 7:
            return None
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"]).set_index("date")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_index()
            if start:
                df = df[df.index >= start]
            print(f"  [FRED {series_id}] using cached data ({age_days:.1f} days old)")
            return df
        except Exception:
            return None

    # -- Primary: pandas_datareader (FRED JSON endpoint) ----------------------
    try:
        import pandas_datareader.data as web
        df_raw = web.DataReader(series_id, "fred", start_dt)
        df = _to_tidy(df_raw)
        if start:
            df = df[df.index >= start]
        return _cache_and_return(df)
    except Exception as exc:
        print(f"  [FRED {series_id}] pandas_datareader failed ({exc.__class__.__name__}) -- trying direct CSV...")

    # -- Fallback: direct CSV download ----------------------------------------
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        if start:
            df = df[df.index >= start]
        return _cache_and_return(df)
    except Exception as exc2:
        print(f"  [FRED {series_id}] direct CSV also failed ({exc2.__class__.__name__})")

    # -- Last resort: local cache (up to 7 days old) --------------------------
    cached = _load_cache()
    if cached is not None:
        return cached

    raise RuntimeError(f"FRED {series_id}: all fetch methods failed and no usable cache")


def _shiller_fred_tail(df):
    """
    Extend the Shiller dataframe past the XLS publication cutoff (~18-24 month lag).
    Operates on a copy of the input so callers never see SettingWithCopyWarning.

    Strategy
    --------
    1. Try FRED series 'CAPE' for any months newer than the file's last entry.
       (FRED mirrors Shiller's data but sometimes updates slightly faster.)
    2. If that yields nothing (same cutoff as XLS), estimate CAPE from the
       S&P 500 price change via yfinance:
           est_CAPE = last_CAPE * (price_now / price_then) / e10_drift
       e10_drift (~1.02/yr) corrects for the 10-yr earnings denominator slowly
       rising as strong recent earnings replace post-2008 recovery years.
    3. Extend GS10 (10-Yr Treasury) from FRED independently -- different
       source, only ~1 month lag, so it is current.

    'estimated' column: False = Shiller XLS data, True = this synthetic tail.
    Returns df unchanged (estimated=False) if both paths fail.
    """
    df = df.copy()
    df["estimated"] = False
    try:
        last_date  = df.index[-1]
        today      = datetime.datetime.today()
        start_str  = (last_date - datetime.timedelta(days=35)).strftime("%Y-%m-%d")

        # -- 1. Try FRED CAPE series -------------------------------------------
        cape_tail = pd.Series(dtype=float)
        try:
            raw   = _fred("CAPE", start=start_str)
            after = raw[raw.index > last_date]
            if len(after) > 0:
                cape_tail = after["value"].rename("cape")
                print(f"  [Shiller] FRED CAPE tail: "
                      f"{cape_tail.index[0].strftime('%b %Y')} -- "
                      f"{cape_tail.index[-1].strftime('%b %Y')} "
                      f"({len(cape_tail)} months)")
        except Exception:
            pass

        # -- 2. Fall back: estimate from S&P 500 price change -----------------
        if len(cape_tail) == 0:
            import yfinance as yf
            gspc = yf.download("^GSPC", start=last_date.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True)
            if len(gspc) < 2:
                return df
            closes        = gspc["Close"].squeeze()   # ensure 1-D Series
            price_ratio   = float(closes.iloc[-1]) / float(closes.iloc[0])
            years_elapsed = (today - last_date.to_pydatetime()).days / 365.25
            # Slow upward drift in the 10-yr real-earnings denominator
            e10_drift  = 1.02 ** years_elapsed
            est_cape   = float(df["cape"].iloc[-1]) * price_ratio / e10_drift
            cape_tail  = pd.Series([est_cape],
                                   index=[pd.Timestamp(today.date())], name="cape")
            print(f"  [Shiller] CAPE estimated (price-scaling): {est_cape:.1f}x  "
                  f"(S&P x{price_ratio:.2f} / E10 drift x{e10_drift:.3f} "
                  f"since {last_date.strftime('%b %Y')})")

        # -- 3. Extend GS10 from FRED (independent source, ~1 month lag) ------
        gs10_tail = pd.Series(dtype=float)
        try:
            raw_gs   = _fred("GS10", start=start_str)
            after_gs = raw_gs[raw_gs.index > last_date]
            if len(after_gs) > 0:
                gs10_tail = after_gs["value"].rename("gs10")
        except Exception:
            pass
        if len(gs10_tail) == 0:
            gs10_tail = pd.Series([float(df["gs10"].iloc[-1])],
                                  index=[cape_tail.index[-1]], name="gs10")

        # -- Align cape + gs10, build tail df ---------------------------------
        tail = (pd.DataFrame({"cape": cape_tail})
                .join(pd.DataFrame({"gs10": gs10_tail}), how="outer"))
        tail["cape"] = tail["cape"].ffill()
        tail["gs10"] = tail["gs10"].ffill()
        tail = tail.dropna(subset=["cape"])

        tail["earnings_yield"] = 100.0 / tail["cape"]
        tail["pe_trailing"]    = np.nan
        tail["price_nom"]      = np.nan
        tail["earn_nom"]       = np.nan
        tail["estimated"]      = True

        out  = pd.concat([df, tail[df.columns]])
        span = (f"{tail.index[0].strftime('%b %Y')} -- "
                f"{tail.index[-1].strftime('%b %Y')}  ({len(tail)} rows)")
        print(f"  [Shiller] estimated tail appended: {span}")
        return out
    except Exception as exc:
        print(f"  [Shiller] tail extension skipped ({exc.__class__.__name__}: {exc})")
        return df


def _buffett_estimated_tail(df):
    """
    Project the Buffett Indicator to approximately today using:
      - S&P 500 price change (yfinance ^GSPC) to scale equity market cap forward.
      - 5 % nominal GDP growth per year for the denominator.
    Adds an 'estimated' boolean column and appends a single projected row for
    today.  Returns df unchanged (with estimated=False) on any failure.
    """
    df = df.copy()
    df["estimated"] = False
    try:
        import yfinance as yf
        last_date   = df.index[-1]
        today       = datetime.datetime.today()
        days_behind = (today - last_date.to_pydatetime()).days
        if days_behind < 45:
            return df   # already nearly current
        last_ratio = float(df["ratio"].iloc[-1])
        gspc = yf.download("^GSPC", start=last_date.strftime("%Y-%m-%d"),
                           progress=False, auto_adjust=True)
        if len(gspc) < 2:
            return df
        closes    = gspc["Close"].squeeze()   # ensure 1-D Series
        mkt_scale = float(closes.iloc[-1]) / float(closes.iloc[0])
        gdp_scale = 1.05 ** (days_behind / 365.25)
        est_ratio = last_ratio * mkt_scale / gdp_scale
        est_df    = pd.DataFrame({"ratio": [est_ratio], "estimated": [True]},
                                 index=[pd.Timestamp(today.date())])
        out = pd.concat([df, est_df])
        print(f"  [Buffett] estimated current: {est_ratio:.0f}%  "
              f"(S&P x{mkt_scale:.2f} / GDP +{(gdp_scale-1)*100:.0f}% "
              f"since {last_date.strftime('%b %Y')})")
        return out
    except Exception as exc:
        print(f"  [Buffett] forward estimate skipped ({exc.__class__.__name__})")
        return df


def fetch_buffett():
    """
    Equity market cap (NCBEILQ027S, millions USD) / GDP (billions USD).
    Converts to a consistent ratio (both in same units) expressed as %.
    """
    mc  = _fred("NCBEILQ027S")   # millions of dollars, quarterly
    gdp = _fred("GDP")            # billions of dollars, annual rate, quarterly
    # Align on quarter ends
    mc  = mc.resample("QE").last()
    gdp = gdp.resample("QE").last()
    merged = mc.join(gdp, lsuffix="_mc", rsuffix="_gdp").dropna()
    # Both in billions for a clean ratio
    merged["ratio"] = (merged["value_mc"] / 1000) / merged["value_gdp"] * 100
    return _buffett_estimated_tail(merged[["ratio"]])


def fetch_shiller():
    """
    Download Robert Shiller's IE dataset from Yale and return
    a DataFrame with columns: cape, gs10, earnings_yield.
    """
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    r   = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data",
                       header=7, engine="xlrd")

    # Date column is decimal year e.g. 1871.01 -> Jan 1871
    def _parse_date(v):
        try:
            v = float(v)
            year  = int(v)
            month = round((v - year) * 100)
            month = max(1, min(12, month))
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception:
            return pd.NaT

    df["date"] = df["Date"].apply(_parse_date)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Keep relevant columns; rename for clarity
    df = df.rename(columns={"Rate GS10": "gs10", "CAPE": "cape",
                             "P": "price_nom", "E": "earn_nom"})
    for col in ("cape", "gs10", "price_nom", "earn_nom"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["cape", "gs10", "price_nom", "earn_nom"]].dropna(subset=["cape"])
    df["earnings_yield"] = 100 / df["cape"]
    # Trailing 12-month P/E = nominal price / nominal trailing earnings
    # Cap to [5, 200] to drop outliers (e.g. earnings near zero during recessions)
    raw_pe = df["price_nom"] / df["earn_nom"]
    df["pe_trailing"] = raw_pe.where((raw_pe > 5) & (raw_pe < 200))
    return _shiller_fred_tail(df)


def fetch_sector_composition(sp500_tickers=None):
    """
    Fetch sector breakdown of the S&P 500 from TradingView.
    Returns a DataFrame with columns:
        sector, weight_pct, pe_sector, contribution, mktcap_bn
    where contribution = weight_pct/100 * pe_sector
    (the number of P/E points that sector adds to the blended index multiple).
    Returns None on failure.
    """
    try:
        from tradingview_screener import Query, col as tvcol
    except ImportError:
        return None
    try:
        tickers = sp500_tickers or _sp500_tickers()
        q = (Query()
             .select("name", "sector", "market_cap_basic", "price_earnings_ttm")
             .set_markets("america"))
        if tickers:
            q = q.where(tvcol("name").isin(tickers))
        else:
            q = q.where(tvcol("market_cap_basic") > 1e9,
                        tvcol("is_primary") == True)   # noqa: E712
        _, df = q.order_by("market_cap_basic", ascending=False).limit(503).get_scanner_data()
        df = df.dropna(subset=["market_cap_basic", "sector"])
        df["market_cap_basic"]    = pd.to_numeric(df["market_cap_basic"],    errors="coerce")
        df["price_earnings_ttm"]  = pd.to_numeric(df["price_earnings_ttm"],  errors="coerce")

        # Drop dual-class secondary tickers to avoid double-counting
        # (TradingView reports full-company market cap per ticker)
        for secondary, (primary, _display) in _DUAL_CLASS.items():
            sec_mask = df["name"] == secondary
            pri_mask = df["name"] == primary
            if sec_mask.any() and pri_mask.any():
                df = df[~sec_mask].copy()

        total_mc = df["market_cap_basic"].sum()

        rows = []
        for sector, grp in df.groupby("sector"):
            mc = float(grp["market_cap_basic"].sum())
            wt = mc / total_mc * 100
            # Market-cap weighted P/E using only profitable stocks (avoid distortion)
            pos = grp[(grp["price_earnings_ttm"] > 3) & (grp["price_earnings_ttm"] < 300)]
            if len(pos) >= 2:
                pe_s = float((pos["price_earnings_ttm"] * pos["market_cap_basic"]).sum()
                             / pos["market_cap_basic"].sum())
            else:
                pe_s = None
            rows.append({
                "sector":       str(sector),
                "weight_pct":   round(wt, 2),
                "pe_sector":    round(pe_s, 1) if pe_s is not None else None,
                "contribution": round(wt * pe_s / 100, 2) if pe_s is not None else 0.0,
                "mktcap_bn":    round(mc / 1e9, 0),
            })

        out = (pd.DataFrame(rows)
               .dropna(subset=["pe_sector"])
               .sort_values("contribution", ascending=False)
               .reset_index(drop=True))
        return out
    except Exception as exc:
        print(f"  [sector] fetch failed: {exc}")
        return None


def _sp500_tickers():
    """
    Return the current S&P 500 constituent ticker list scraped from Wikipedia.
    Uses requests with a browser User-Agent to avoid 403.
    Falls back to an empty list on failure (caller will use a large-cap filter).
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-tool/1.0)"},
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        syms = tables[0]["Symbol"].astype(str).str.strip().tolist()
        # Wikipedia uses '.' (BRK.B); TradingView also uses '.'
        return [s for s in syms if s]
    except Exception as e:
        print(f"  [concentration] Wikipedia S&P 500 list unavailable ({e}) -- using large-cap fallback")
        return []


# Dual-class share pairs in the S&P 500.
# Key = secondary ticker, value = (primary ticker, display name).
# When both appear, the secondary is merged into the primary row.
_DUAL_CLASS = {
    "GOOG":  ("GOOGL", "Alphabet Inc."),
    "NWS":   ("NWSA",  "News Corp"),
    "FOX":   ("FOXA",  "Fox Corp"),
}


def fetch_concentration():
    """
    Compute current top-10 stock weight in the S&P 500 from
    TradingView screener (market cap per constituent).

    Dual-class share pairs (e.g. GOOGL + GOOG) are merged into a single
    company entry before slicing the top 10, so one company does not occupy
    two slots.

    Returns (top10_pct, list of (ticker, name, weight_pct, mktcap_bn)).
    """
    try:
        from tradingview_screener import Query, col as tvcol
    except ImportError:
        print("  [concentration] tradingview-screener not installed -- skipping live fetch")
        return None, []

    try:
        tickers = _sp500_tickers()
        q = Query().select("name", "market_cap_basic", "description").set_markets("america")

        if tickers:
            q = q.where(tvcol("name").isin(tickers))
        else:
            q = q.where(
                tvcol("market_cap_basic") > 1e9,
                tvcol("is_primary") == True,   # noqa: E712
            )

        _, df = (q.order_by("market_cap_basic", ascending=False)
                  .limit(503)
                  .get_scanner_data())

        df = df.dropna(subset=["market_cap_basic"])

        # ── Deduplicate dual-class share pairs ───────────────────────────────
        # TradingView reports the TOTAL company market cap for every ticker,
        # regardless of share class.  Adding secondary mc to primary would
        # double-count the company.  Just drop the secondary row.
        for secondary, (primary, display_name) in _DUAL_CLASS.items():
            sec_mask = df["name"] == secondary
            pri_mask = df["name"] == primary
            if sec_mask.any() and pri_mask.any():
                df.loc[pri_mask, "description"] = display_name
                df = df[~sec_mask].copy()
            elif sec_mask.any():
                # Primary absent -- promote secondary to canonical ticker
                df.loc[sec_mask, "name"]        = primary
                df.loc[sec_mask, "description"] = display_name

        # Re-sort after potential market-cap changes from merging
        df = df.sort_values("market_cap_basic", ascending=False).reset_index(drop=True)

        total = df["market_cap_basic"].sum()
        df["weight"] = df["market_cap_basic"] / total * 100
        top10 = df.head(10)
        top10_pct = float(top10["weight"].sum())

        holdings = []
        for _, row in top10.iterrows():
            holdings.append((
                str(row.get("name", "")),
                str(row.get("description", row.get("name", ""))),
                float(row["weight"]),
                float(row["market_cap_basic"]) / 1e9,
            ))
        return top10_pct, holdings

    except Exception as e:
        print(f"  [concentration] fetch failed: {e}")
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    """Render a matplotlib figure to a base-64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _base_fig(w=12, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(S1)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.xaxis.label.set_color(SUB)
    ax.yaxis.label.set_color(SUB)
    for spine in ax.spines.values():
        spine.set_color(BD)
    ax.grid(axis="y", color=BD, linewidth=0.6, zorder=0)
    ax.grid(axis="x", color="#111420", linewidth=0.4, zorder=0)
    return fig, ax


def _band(ax, y_lo, y_hi, color, alpha=0.12, label=None):
    ax.axhspan(y_lo, y_hi, color=color, alpha=alpha, zorder=0, label=label)


def _mean_sd_bands(ax, series, color=LINE1):
    m  = series.mean()
    sd = series.std()
    ax.axhline(m,       color=color,  lw=1.0, ls="--", alpha=0.7, label=f"Mean {m:.1f}")
    ax.axhline(m + sd,  color=YLW,    lw=0.7, ls=":",  alpha=0.6, label=f"+1 SD {m+sd:.1f}")
    ax.axhline(m - sd,  color=GRN,    lw=0.7, ls=":",  alpha=0.6, label=f"-1 SD {m-sd:.1f}")


def _extend_to_today(ax, series, color, threshold_days=90):
    """
    Push the x-axis right edge to today + 6-month margin.
    When the last data point is more than threshold_days before today, also
    draw a flat dotted line + label so the publication lag is visible.
    When estimated data already reaches near today the xlim is updated but
    no redundant extension line is drawn.
    """
    today     = datetime.datetime.today()
    last_date = series.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()
    ax.set_xlim(right=today + datetime.timedelta(days=200))
    days_behind = (today - last_date).days
    if days_behind < threshold_days:
        return   # data is already recent -- no stale-data annotation needed
    last_val  = float(series.iloc[-1])
    ax.plot([last_date, today], [last_val, last_val],
            color=color, lw=1.0, ls=":", alpha=0.45, zorder=2)
    last_str = series.index[-1].strftime("%b %Y")
    ax.annotate(f"last data: {last_str}",
                xy=(last_date, last_val),
                xytext=(6, 6), textcoords="offset points",
                color=SUB, fontsize=7, style="italic")


def chart_concentration(hist_data, current_pct, holdings):
    fig, ax = _base_fig(h=4.2)

    # Historical anchor points
    hist_years = [y for y, _ in hist_data]
    hist_vals  = [v for _, v in hist_data]
    hist_dates = [datetime.date(y, 6, 30) for y in hist_years]

    ax.plot(hist_dates, hist_vals, color=LINE1, lw=1.5, alpha=0.6,
            ls="--", marker="o", ms=4, label="Historical (anchor pts)")

    # Live current point
    if current_pct is not None:
        today = datetime.date.today()
        # Connect last historical anchor to today
        last_hist_date = hist_dates[-1]
        last_hist_val  = hist_vals[-1]
        ax.plot([last_hist_date, today], [last_hist_val, current_pct],
                color=LINE1, lw=2.0, alpha=0.9)
        ax.scatter([today], [current_pct], color=LINE1, s=60, zorder=5,
                   label=f"Live ({today.year}): {current_pct:.1f}%")

    # Mean lines on anchored data
    all_vals = hist_vals + ([current_pct] if current_pct else [])
    mean_v = np.mean(all_vals)
    ax.axhline(mean_v, color=SUB, lw=0.8, ls="--", alpha=0.5,
               label=f"Mean {mean_v:.1f}%")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("Top-10 weight (%)", color=SUB, fontsize=9)
    ax.set_title("S&P 500 Top-10 Concentration  (% of index market cap)",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD,
              labelcolor=SUB, loc="upper left")
    # Always extend x-axis to today regardless of data availability
    ax.set_xlim(right=datetime.datetime.today() + datetime.timedelta(days=200))

    # Source note
    fig.text(0.01, 0.01,
             "Sources: TradingView (live), Mauboussin/Credit Suisse, Goldman Sachs, S&P DJI (historical anchors)",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def chart_buffett(df_buffett):
    fig, ax = _base_fig(h=4.0)

    # Split FRED quarterly data (solid) vs yfinance-projected point (dotted)
    s_full = df_buffett["ratio"].dropna()
    if "estimated" in df_buffett.columns:
        est_mask = df_buffett["estimated"].reindex(s_full.index).fillna(False)
        s_real   = s_full[~est_mask]
        s_est    = s_full[est_mask]
    else:
        s_real = s_full
        s_est  = pd.Series(dtype=float)

    mean_v = s_real.mean()
    sd_v   = s_real.std()

    # Valuation bands (relative to real-data mean)
    ax.axhspan(0,              mean_v - sd_v,   color=GRN, alpha=0.07)
    ax.axhspan(mean_v - sd_v,  mean_v,          color=GRN, alpha=0.04)
    ax.axhspan(mean_v,         mean_v + sd_v,   color=YLW, alpha=0.07)
    ax.axhspan(mean_v + sd_v,  s_real.max()*1.1, color=RED, alpha=0.07)

    ax.plot(s_real.index.to_pydatetime(), s_real.values, color=LINE2, lw=1.8, zorder=3)
    _mean_sd_bands(ax, s_real, color=LINE2)

    # Estimated current point: dotted bridge from last FRED entry + diamond marker
    if len(s_est) > 0:
        last_real_dt = s_real.index[-1].to_pydatetime()
        last_real_v  = float(s_real.iloc[-1])
        est_dt       = s_est.index[-1].to_pydatetime()
        est_v        = float(s_est.iloc[-1])
        ax.plot([last_real_dt, est_dt], [last_real_v, est_v],
                color=LINE2, lw=1.5, ls=":", alpha=0.6, zorder=3)
        ax.scatter([est_dt], [est_v], color=LINE2, s=80, zorder=5, marker="D",
                   label=f"Est. today: {est_v:.0f}%")
        cur_dt, cur = est_dt, est_v
    else:
        cur_dt = s_real.index[-1].to_pydatetime()
        cur    = float(s_real.iloc[-1])
        ax.scatter([cur_dt], [cur], color=LINE2, s=50, zorder=5)

    ax.annotate(f"{cur:.0f}%", xy=(cur_dt, cur),
                xytext=(-46, 8), textcoords="offset points",
                color=TEXT, fontsize=9, fontweight="bold")

    _extend_to_today(ax, s_full, LINE2)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("Equity Mkt Cap / GDP (%)", color=SUB, fontsize=9)
    ax.set_title("Buffett Indicator  --  Equity Market Cap / GDP",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD,
              labelcolor=SUB, loc="upper left")

    legend_patches = [
        Patch(color=GRN, alpha=0.35, label="Below mean (cheap)"),
        Patch(color=YLW, alpha=0.35, label="Mean to +1 SD"),
        Patch(color=RED, alpha=0.35, label="Above +1 SD (extended)"),
    ]
    ax2_leg = ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.2,
                        facecolor=S2, edgecolor=BD, labelcolor=SUB, loc="upper left")
    ax.add_artist(ax2_leg)

    fig.text(0.01, 0.01,
             "Source: FRED NCBEILQ027S (nonfinancial equity market value) / GDP.  Quarterly, 1945-present.",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def chart_cape(df_shiller):
    fig, ax = _base_fig(h=4.0)

    # Split Shiller-file (solid) vs FRED-tail (estimated, dotted)
    s_full = df_shiller["cape"].dropna()
    if "estimated" in df_shiller.columns:
        est_mask = df_shiller["estimated"].reindex(s_full.index).fillna(False)
        s_real   = s_full[~est_mask]
        s_est    = s_full[est_mask]
    else:
        s_real = s_full
        s_est  = pd.Series(dtype=float)

    mean_v = s_real.mean()
    sd_v   = s_real.std()

    _band(ax, 0,           mean_v - sd_v, GRN)
    _band(ax, mean_v - sd_v, mean_v,      GRN, alpha=0.05)
    _band(ax, mean_v,      mean_v + sd_v, YLW)
    _band(ax, mean_v + sd_v, s_real.max()*1.1, RED)

    # Solid line: Shiller XLS data
    ax.plot(s_real.index.to_pydatetime(), s_real.values,
            color=LINE3, lw=1.5, zorder=3, label="Shiller CAPE (file)")
    _mean_sd_bands(ax, s_real, color=LINE3)

    # Dotted tail: FRED CAPE (fills the publication-lag gap)
    if len(s_est) > 0:
        bridge_x = [s_real.index[-1].to_pydatetime(), s_est.index[0].to_pydatetime()]
        bridge_y = [float(s_real.iloc[-1]), float(s_est.iloc[0])]
        ax.plot(bridge_x, bridge_y, color=LINE3, lw=1.2, ls=":", alpha=0.55, zorder=3)
        ax.plot(s_est.index.to_pydatetime(), s_est.values,
                color=LINE3, lw=1.5, ls=":", alpha=0.75, zorder=3,
                label="CAPE (FRED -- current)")
        cur_date = s_est.index[-1].to_pydatetime()
        cur      = float(s_est.iloc[-1])
        ax.scatter([cur_date], [cur], color=LINE3, s=60, zorder=5, marker="D")
    else:
        cur_date = s_real.index[-1].to_pydatetime()
        cur      = float(s_real.iloc[-1])
        ax.scatter([cur_date], [cur], color=LINE3, s=50, zorder=5)

    ax.annotate(f"{cur:.1f}x", xy=(cur_date, cur),
                xytext=(-44, 8), textcoords="offset points",
                color=TEXT, fontsize=9, fontweight="bold")

    _extend_to_today(ax, s_full, LINE3)

    ax.set_ylabel("CAPE (Shiller P/E)", color=SUB, fontsize=9)
    ax.set_title("Shiller CAPE  --  Cyclically Adjusted P/E Ratio",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD,
              labelcolor=SUB, loc="upper left")

    fig.text(0.01, 0.01,
             "Source: Robert Shiller, Yale University (ie_data.xls) + FRED CAPE series.  Monthly, 1871-present.",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def chart_fed_model(df_shiller):
    """Earnings yield (1/CAPE) minus 10-Y Treasury yield = Equity Risk Premium proxy."""
    # Preserve 'estimated' column if present
    keep = ["earnings_yield", "gs10"]
    if "estimated" in df_shiller.columns:
        keep.append("estimated")
    df = df_shiller[keep].dropna(subset=["earnings_yield", "gs10"]).copy()
    df = df[df.index >= "1953-01-01"]
    df["erp"] = df["earnings_yield"] - df["gs10"]

    # Split real vs estimated
    has_est = "estimated" in df.columns
    if has_est:
        df_real = df[~df["estimated"].fillna(False)]
        df_est  = df[df["estimated"].fillna(False)]
    else:
        df_real = df
        df_est  = df.iloc[0:0]

    fig, ax = _base_fig(h=3.8)

    # Filled area + solid line: real (Shiller XLS) data
    d_real = df_real.index.to_pydatetime()
    e_real = df_real["erp"].values
    ax.fill_between(d_real, e_real, 0,
                    where=(e_real >= 0), color=GRN, alpha=0.25, label="Equities cheap vs bonds")
    ax.fill_between(d_real, e_real, 0,
                    where=(e_real <  0), color=RED, alpha=0.20, label="Equities expensive vs bonds")
    ax.plot(d_real, e_real, color=LINE1, lw=1.2, zorder=3)
    ax.axhline(0, color=SUB, lw=0.8, ls="--", alpha=0.5)

    # Dotted tail: estimated (FRED CAPE + GS10) data
    if len(df_est) > 0:
        d_est = df_est.index.to_pydatetime()
        e_est = df_est["erp"].values
        # Bridge from last real point
        ax.plot([d_real[-1], d_est[0]], [float(df_real["erp"].iloc[-1]), e_est[0]],
                color=LINE1, lw=1.0, ls=":", alpha=0.5, zorder=3)
        ax.plot(d_est, e_est, color=LINE1, lw=1.2, ls=":", alpha=0.7, zorder=3,
                label="ERP (FRED -- current)")
        cur_dt = d_est[-1]
        cur    = float(df_est["erp"].iloc[-1])
    else:
        cur_dt = d_real[-1]
        cur    = float(df_real["erp"].iloc[-1])

    ax.scatter([cur_dt], [cur], color=LINE1, s=50, zorder=5)
    direction = "cheap vs bonds" if cur > 0 else "expensive vs bonds"
    ax.annotate(f"{cur:+.2f}%  ({direction})", xy=(cur_dt, cur),
                xytext=(-120, 10), textcoords="offset points",
                color=TEXT, fontsize=9, fontweight="bold")

    _extend_to_today(ax, df["erp"], LINE1)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
    ax.set_ylabel("Earnings Yield - 10Y Treasury (%)", color=SUB, fontsize=9)
    ax.set_title("Fed Model  --  Equity Earnings Yield minus 10-Year Treasury Yield",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD,
              labelcolor=SUB, loc="lower left")

    fig.text(0.01, 0.01,
             "Earnings yield = 100 / Shiller CAPE.  Treasury yield: FRED GS10.  "
             "Dotted tail = FRED CAPE + GS10 estimate (current).  Monthly, 1953-present.",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def chart_pe_context(df_shiller, live_pe=None):
    """
    CAPE vs trailing 12-month P/E since 1990.
    X-axis is explicitly extended to today so the live P/E diamond is always
    visible even when Shiller data has an 18-24 month lag.
    A dotted bridge connects the last Shiller trailing-P/E point to the diamond.
    """
    today  = datetime.datetime.today()
    df     = df_shiller[df_shiller.index >= "1990-01-01"].copy()
    fig, ax = _base_fig(h=4.2)

    cape_s  = df["cape"].dropna()
    pe_s    = df["pe_trailing"].dropna() if "pe_trailing" in df.columns else pd.Series(dtype=float)

    # -- Shade gap between CAPE and trailing P/E where they overlap -----------
    if len(pe_s) and len(cape_s):
        aligned = pd.concat([pe_s.rename("pe"), cape_s.rename("cape")],
                            axis=1).dropna()
        if len(aligned):
            ax.fill_between(aligned.index.to_pydatetime(),
                            aligned["pe"], aligned["cape"],
                            where=(aligned["cape"] > aligned["pe"]),
                            color=LINE3, alpha=0.12, zorder=1,
                            label="CAPE premium over trailing P/E")

    # -- Trailing P/E line (Shiller; ends ~18-24 months ago) -----------------
    if len(pe_s):
        dates_pe = pe_s.index.to_pydatetime()
        ax.plot(dates_pe, pe_s.values, color=LINE2, lw=1.8,
                label="Trailing P/E (12-month earnings)", zorder=3)
        pe_mean = float(pe_s.mean())
        ax.axhline(pe_mean, color=LINE2, lw=0.9, ls="--", alpha=0.55,
                   label=f"Trailing P/E mean  {pe_mean:.1f}x")
        # Dotted bridge from last Shiller point to live diamond
        if live_pe is not None:
            ax.plot([dates_pe[-1], today],
                    [float(pe_s.iloc[-1]), live_pe],
                    color=LINE2, lw=1.2, ls=":", alpha=0.55, zorder=2)

    # -- Live blended P/E diamond (TradingView, today) -----------------------
    if live_pe is not None:
        ax.scatter([today], [live_pe], color=LINE2, s=110, zorder=7,
                   marker="D", label=f"Live blended P/E  {live_pe:.1f}x")
        ax.annotate(f"{live_pe:.1f}x  (live)",
                    xy=(today, live_pe),
                    xytext=(-82, 12), textcoords="offset points",
                    color=LINE2, fontsize=9, fontweight="bold")

    # -- CAPE line -----------------------------------------------------------
    dates_cape = cape_s.index.to_pydatetime()
    ax.plot(dates_cape, cape_s.values, color=LINE3, lw=1.8,
            label="Shiller CAPE (10-yr avg earnings)", zorder=3)
    cape_mean = float(cape_s.mean())
    ax.axhline(cape_mean, color=LINE3, lw=0.9, ls="--", alpha=0.55,
               label=f"CAPE mean  {cape_mean:.1f}x")

    cur_cape = float(cape_s.iloc[-1])
    ax.scatter([dates_cape[-1]], [cur_cape], color=LINE3, s=60, zorder=5)
    ax.annotate(f"{cur_cape:.1f}x  (CAPE)",
                xy=(dates_cape[-1], cur_cape),
                xytext=(-88, -17), textcoords="offset points",
                color=TEXT, fontsize=9, fontweight="bold")

    # -- Explicit x-axis: 1990 to today + 12-month right margin -------------
    ax.set_xlim(
        left  = datetime.datetime(1990, 1, 1),
        right = today + datetime.timedelta(days=400),
    )

    ax.set_ylabel("Price / Earnings multiple (x)", color=SUB, fontsize=9)
    ax.set_title("Valuation Normalization  --  CAPE vs Trailing P/E (1990-present)",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD,
              labelcolor=SUB, loc="upper left", ncol=2)

    fig.text(0.01, 0.01,
             "CAPE uses 10-yr inflation-adj avg earnings -- penalises recovery years after recessions.  "
             "Trailing P/E = price / current 12M earnings.  Dotted line = bridge to live TradingView reading.  "
             "Source: Shiller/Yale + TradingView.",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


# Sector colour palette (consistent, dark-theme friendly)
_SECTOR_COLORS = {
    "Technology":               "#6366f1",   # indigo  -- accent
    "Communication Services":   "#a78bfa",   # violet
    "Consumer Discretionary":   "#60a5fa",   # sky blue (AMZN, TSLA)
    "Health Care":              "#34d399",   # emerald
    "Financials":               "#f59e0b",   # amber
    "Industrials":              "#94a3b8",   # slate
    "Energy":                   "#fb923c",   # orange
    "Consumer Staples":         "#9ca3af",   # cool gray
    "Materials":                "#6b7280",   # gray
    "Real Estate":              "#4b5563",   # dark gray
    "Utilities":                "#374151",   # darkest gray
}


def chart_sector_decomp(df_sectors):
    """
    Horizontal bar chart: each sector's contribution to the blended index P/E.
    Contribution (pts) = (sector_weight% / 100) * sector_weighted_P/E.
    Summing all sectors gives the blended index trailing P/E.

    A vertical line at the historical mean P/E (~17x) provides context:
    the total bar length vs that line shows how far above normal the market is.
    A second annotation shows what the blended multiple would be if the two
    highest-multiple sectors (Technology + Communication Services) had the
    same P/E as the rest of the market.
    """
    if df_sectors is None or len(df_sectors) == 0:
        return None

    # Sort ascending so largest bar is at the top when plotted
    df = df_sectors.sort_values("contribution", ascending=True).reset_index(drop=True)
    blended = float(df["contribution"].sum())
    HIST_MEAN_PE = 17.0   # long-run S&P 500 trailing P/E mean (approx)

    # Counterfactual: if Tech + Comms had same P/E as rest of market
    high_mult = ["Technology", "Communication Services"]
    rest = df[~df["sector"].isin(high_mult)]
    rest_wt  = rest["weight_pct"].sum()
    rest_contrib = rest["contribution"].sum()
    rest_avg_pe  = (rest_contrib / (rest_wt / 100)) if rest_wt > 0 else blended
    hm_wt = df[df["sector"].isin(high_mult)]["weight_pct"].sum()
    counterfactual = rest_contrib + (hm_wt / 100 * rest_avg_pe)

    n = len(df)
    fig, ax = _base_fig(h=max(3.8, n * 0.52 + 1.2))

    colors = [_SECTOR_COLORS.get(s, "#64748b") for s in df["sector"]]
    bars = ax.barh(range(n), df["contribution"].values,
                   color=colors, height=0.62, zorder=3)

    # Sector labels on y-axis
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["sector"].tolist(), color=TEXT, fontsize=9)

    # Right-side annotations: "X.Xpts  (YY% wt, ZZx PE)"
    x_max = df["contribution"].max() * 1.55
    for i, row in df.iterrows():
        ax.text(float(row["contribution"]) + blended * 0.02, i,
                f"{row['contribution']:.1f} pts  "
                f"({row['weight_pct']:.0f}% wt, {row['pe_sector']:.0f}x PE)",
                va="center", color=SUB, fontsize=8)

    # Blended total line
    ax.axvline(blended, color=TEXT, lw=1.1, ls="--", alpha=0.75, zorder=4)
    ax.text(blended + blended * 0.01, n - 0.55,
            f"Blended: {blended:.1f}x", color=TEXT, fontsize=8.5, fontweight="bold")

    # Historical mean line
    ax.axvline(HIST_MEAN_PE, color=GRN, lw=1.0, ls=":", alpha=0.8, zorder=4)
    ax.text(HIST_MEAN_PE - blended * 0.01, -0.75,
            f"Hist. avg ~{HIST_MEAN_PE:.0f}x", color=GRN, fontsize=7.5, ha="right")

    # Counterfactual annotation
    ax.annotate(
        f"If Tech + Comms had market-avg PE ({rest_avg_pe:.0f}x): blended ~{counterfactual:.1f}x",
        xy=(0, -0.55), xycoords=("data", "data"),
        color=SUB, fontsize=8, style="italic",
    )

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Contribution to blended index P/E (pts)", color=SUB, fontsize=9)
    ax.set_title("Sector P/E Contribution to Blended S&P 500 Multiple",
                 color=TEXT, fontsize=11, pad=10, loc="left")
    ax.tick_params(axis="x", colors=SUB, labelsize=8)

    fig.text(0.01, 0.01,
             "Each bar = (sector weight% / 100) x sector market-cap-weighted P/E.  "
             "Bars sum to blended index P/E.  Profitable companies only.  Source: TradingView.",
             color=SUB, fontsize=7, va="bottom")
    plt.tight_layout(pad=1.4)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Help modal
# ─────────────────────────────────────────────────────────────────────────────
_HELP_CSS = """
.help-overlay{display:none;position:fixed;inset:0;z-index:9990;background:rgba(0,0,0,.65);backdrop-filter:blur(4px)}
.help-overlay.open{display:flex;align-items:center;justify-content:center}
.help-modal{background:#1a1e2e;border:1px solid #2d3348;border-radius:12px;width:660px;max-width:95vw;max-height:85vh;overflow-y:auto;padding:28px 32px;position:relative;box-shadow:0 24px 64px rgba(0,0,0,.6)}
.help-modal h2{font-size:16px;font-weight:700;color:#e2e8f0;margin:0 0 8px}
.help-desc{font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:16px}
.help-sec{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin:18px 0 8px;border-bottom:1px solid #252a3a;padding-bottom:6px;font-weight:700}
.help-tbl{width:100%;border-collapse:collapse;font-size:12px}
.help-tbl td{padding:6px 8px;border-bottom:1px solid #1e2234;vertical-align:top}
.help-tbl td:first-child{color:#e2e8f0;font-weight:600;white-space:nowrap;min-width:150px;padding-right:14px}
.help-tbl td:last-child{color:#94a3b8;line-height:1.6}
.help-close{position:absolute;top:14px;right:14px;background:none;border:1px solid #2d3348;border-radius:6px;color:#94a3b8;font-size:14px;cursor:pointer;padding:3px 10px}
.help-close:hover{color:#e2e8f0;border-color:#6366f1}
.help-btn{display:inline-flex;align-items:center;gap:5px;padding:5px 13px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.35);color:#a5b4fc;cursor:pointer;transition:opacity .15s;white-space:nowrap}
.help-btn:hover{opacity:.8}
"""

_HELP_JS = """
function openHelp(){document.getElementById('helpOverlay').classList.add('open')}
function closeHelp(){document.getElementById('helpOverlay').classList.remove('open')}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeHelp()});
"""

_HELP_MODAL = """
<div class="help-overlay" id="helpOverlay" onclick="if(event.target===this)closeHelp()">
 <div class="help-modal">
  <button class="help-close" onclick="closeHelp()">&#x2715;</button>
  <h2>Market Concentration &amp; Macro Valuation</h2>
  <p class="help-desc">
   Four long-run gauges of market valuation and structure. Each chart shows the
   current reading alongside its full historical range so you can judge how
   stretched (or cheap) conditions are relative to history.
   Green bands = below historical mean; yellow = mean to +1 SD; red = above +1 SD.
  </p>
  <div class="help-sec">Indicators</div>
  <table class="help-tbl">
   <tr><td>Concentration</td>
       <td>Top-10 stocks as % of S&P 500 total market cap. Higher = more index returns
           driven by a handful of names. Historically peaks near market tops.</td></tr>
   <tr><td>Buffett Indicator</td>
       <td>Total equity market cap divided by nominal GDP. Warren Buffett called it
           "probably the best single measure" of overall market valuation. Current
           reading uses FRED nonfinancial corporate equity / GDP (1945-present).
           Mean and standard-deviation bands provide context.</td></tr>
   <tr><td>Shiller CAPE</td>
       <td>S&P 500 price divided by the 10-year average of real (inflation-adjusted)
           earnings. Smooths out the earnings cycle. Data from Robert Shiller (Yale),
           monthly back to 1871. Long-run mean ~17x; readings above 30x have historically
           preceded below-average 10-year returns.</td></tr>
   <tr><td>Fed Model (ERP)</td>
       <td>S&P 500 earnings yield (1/CAPE) minus the 10-Year Treasury yield.
           Positive = equities offer more yield than bonds (cheap vs bonds).
           Negative = bonds offer more yield than stocks (stocks expensive vs bonds).</td></tr>
  </table>
  <div class="help-sec">Data Sources</div>
  <table class="help-tbl">
   <tr><td>Concentration (live)</td><td>TradingView screener -- real-time S&P 500 market caps</td></tr>
   <tr><td>Concentration (history)</td><td>Mauboussin / Credit Suisse 2017, Goldman Sachs, S&P DJI</td></tr>
   <tr><td>Buffett Indicator</td><td>FRED: NCBEILQ027S + GDP.  Quarterly 1945-present.</td></tr>
   <tr><td>Shiller CAPE + GS10</td><td>Robert Shiller (Yale): ie_data.xls.  Monthly 1871-present.</td></tr>
  </table>
 </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembler
# ─────────────────────────────────────────────────────────────────────────────
def build_html(charts, summary, holdings, ts, suite_port=5050):
    conc_chart, buffett_chart, cape_chart, fed_chart = charts[:4]
    pe_ctx_chart   = charts[4] if len(charts) > 4 else None
    sector_chart   = charts[5] if len(charts) > 5 else None
    cur_conc, cur_buffett, cur_cape, cur_erp = summary

    def _stat(label, value, note="", color=TEXT):
        return f"""
<div class="stat">
  <div class="stat-val" style="color:{color}">{value}</div>
  <div class="stat-lbl">{label}</div>
  {f'<div class="stat-note">{note}</div>' if note else ''}
</div>"""

    def _verdict(v, mean, sd, labels=("Cheap","Fair","Elevated","Extended")):
        if v is None:
            return labels[1], SUB
        if v < mean - sd:   return labels[0], GRN
        if v < mean:        return labels[1], LINE2
        if v < mean + sd:   return labels[2], YLW
        return labels[3], RED

    # Verdict labels
    b_v, b_c = _verdict(cur_buffett[0], cur_buffett[1], cur_buffett[2])
    c_v, c_c = _verdict(cur_cape[0],    cur_cape[1],    cur_cape[2])

    erp_label = ("Stocks cheap vs bonds" if (cur_erp or 0) > 0
                 else "Stocks expensive vs bonds")
    erp_color = GRN if (cur_erp or 0) > 0 else RED

    # Holdings table rows
    holdings_rows = ""
    for i, (ticker, name, wt, mc) in enumerate(holdings, 1):
        bar = int(wt / max(h[2] for h in holdings) * 120) if holdings else 0
        holdings_rows += f"""
<tr>
  <td style="color:{SUB};font-size:11px">{i}</td>
  <td style="font-weight:700;color:{TEXT}">{ticker}</td>
  <td style="color:{SUB};font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{name}</td>
  <td style="text-align:right;font-weight:600;color:{LINE1}">{wt:.2f}%</td>
  <td style="text-align:right;color:{SUB}">${mc:,.0f}B</td>
  <td><div style="height:6px;border-radius:3px;background:{LINE1};opacity:.7;width:{bar}px"></div></td>
</tr>"""

    conc_section = ""
    if conc_chart:
        conc_section = f"""
<div class="section">
  <div class="sec-hdr">
    <span class="sec-title">S&P 500 Market Concentration</span>
    <span class="sec-sub">Top-10 stocks as % of index market cap &bull; Live + historical anchors back to 1927</span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{conc_chart}" alt="concentration chart">
  {f'''<div class="holdings-wrap">
    <div class="sec-title" style="font-size:11px;margin-bottom:10px;color:{SUB}">CURRENT TOP-10 HOLDINGS</div>
    <table class="h-tbl"><tbody>{holdings_rows}</tbody></table>
  </div>''' if holdings else ''}
</div>"""

    buffett_section = ""
    if buffett_chart:
        buffett_section = f"""
<div class="section">
  <div class="sec-hdr">
    <span class="sec-title">Buffett Indicator</span>
    <span class="sec-sub">Equity market cap / nominal GDP &bull; FRED data 1945-present</span>
    <span class="verdict-badge" style="background:rgba(99,102,241,.12);color:{b_c};border:1px solid {b_c}40">{b_v}</span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{buffett_chart}" alt="buffett indicator chart">
</div>"""

    cape_section = ""
    if cape_chart:
        cape_section = f"""
<div class="section">
  <div class="sec-hdr">
    <span class="sec-title">Shiller CAPE</span>
    <span class="sec-sub">Cyclically adjusted P/E &bull; Shiller / Yale data 1871-present</span>
    <span class="verdict-badge" style="background:rgba(99,102,241,.12);color:{c_c};border:1px solid {c_c}40">{c_v}</span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{cape_chart}" alt="CAPE chart">
</div>"""

    fed_section = ""
    if fed_chart:
        fed_section = f"""
<div class="section">
  <div class="sec-hdr">
    <span class="sec-title">Fed Model  --  Equity Risk Premium</span>
    <span class="sec-sub">Earnings yield (1/CAPE) minus 10-Year Treasury yield &bull; 1953-present</span>
    <span class="verdict-badge" style="background:rgba(99,102,241,.12);color:{erp_color};border:1px solid {erp_color}40">{erp_label}</span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{fed_chart}" alt="fed model chart">
</div>"""

    css = f"""
:root{{--bg:{BG};--s1:{S1};--s2:{S2};--bd:{BD};--text:{TEXT};--sub:{SUB};--acc:#6366f1;--acc2:#818cf8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}}
.nav{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 32px;background:var(--s1);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:200}}
.nav-title{{font-size:13px;font-weight:700;color:var(--text)}}
.hero{{padding:28px 32px 22px;background:linear-gradient(135deg,{BG} 0%,{S1} 60%,{S2} 100%);border-bottom:1px solid var(--bd)}}
.hero h1{{font-size:22px;font-weight:800;color:var(--text);margin-bottom:4px}}
.hero .sub{{font-size:12px;color:var(--sub)}}
.stats{{display:flex;gap:28px;margin-top:18px;flex-wrap:wrap}}
.stat{{display:flex;flex-direction:column;gap:3px;min-width:120px}}
.stat-val{{font-size:22px;font-weight:800}}
.stat-lbl{{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px}}
.stat-note{{font-size:10px;color:var(--sub)}}
.main{{padding:24px 32px 48px;display:flex;flex-direction:column;gap:32px}}
.section{{background:var(--s1);border:1px solid var(--bd);border-radius:12px;overflow:hidden}}
.sec-hdr{{padding:16px 20px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.sec-title{{font-size:13px;font-weight:700;color:var(--text)}}
.sec-sub{{font-size:11px;color:var(--sub)}}
.verdict-badge{{font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px;margin-left:auto}}
.chart-img{{width:100%;display:block}}
.holdings-wrap{{padding:14px 20px 18px;border-top:1px solid var(--bd)}}
.h-tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.h-tbl td{{padding:6px 10px;border-bottom:1px solid {BD}}}
.h-tbl tr:last-child td{{border-bottom:none}}
.footer{{padding:16px 32px;border-top:1px solid var(--bd);font-size:11px;color:var(--sub)}}
@media(max-width:700px){{.main,.hero,.nav{{padding-left:14px;padding-right:14px}}}}
"""

    # ── Valuation context sections ────────────────────────────────────────────
    pe_ctx_section = ""
    if pe_ctx_chart:
        pe_ctx_section = f"""
<div class="section" style="margin-top:4px">
  <div class="sec-hdr">
    <span class="sec-title">Valuation Context  --  CAPE vs Trailing P/E</span>
    <span class="sec-sub">
      CAPE uses 10-yr avg earnings and penalises recovery years after recessions &bull;
      Trailing P/E uses current 12-month earnings (lower = less alarming)
    </span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{pe_ctx_chart}" alt="CAPE vs trailing PE chart">
</div>"""

    sector_section = ""
    if sector_chart:
        sector_section = f"""
<div class="section" style="margin-top:4px">
  <div class="sec-hdr">
    <span class="sec-title">Sector P/E Decomposition</span>
    <span class="sec-sub">
      How many index P/E points each sector contributes &bull;
      Technology &amp; Communication Services account for the bulk of the elevated multiple
    </span>
  </div>
  <img class="chart-img" src="data:image/png;base64,{sector_chart}" alt="sector PE decomposition chart">
</div>"""

    conc_val_s = (f"{cur_conc:.1f}%" if cur_conc is not None else "--")
    buffett_val_s = (f"{cur_buffett[0]:.0f}%" if cur_buffett[0] is not None else "--")
    cape_val_s = (f"{cur_cape[0]:.1f}x" if cur_cape[0] is not None else "--")
    erp_val_s  = (f"{cur_erp:+.2f}%" if cur_erp is not None else "--")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Concentration &amp; Macro Valuation &mdash; {ts}</title>
<style>{css}{_HELP_CSS}</style>
</head>
<body>
<nav class="nav">
  <span class="nav-title">Market Concentration &amp; Macro Valuation</span>
  <button class="help-btn" onclick="openHelp()">&#x24D8; How it works</button>
</nav>
<div class="hero">
  <h1>Market Concentration &amp; Macro Valuation</h1>
  <div class="sub">Generated {ts} &bull; Four long-run valuation gauges + valuation context</div>
  <div class="stats">
    {_stat("Top-10 Concentration", conc_val_s, "% of S&P 500 mkt cap", LINE1)}
    {_stat("Buffett Indicator", buffett_val_s, b_v, b_c)}
    {_stat("Shiller CAPE", cape_val_s, c_v, c_c)}
    {_stat("Equity Risk Premium", erp_val_s, erp_label, erp_color)}
  </div>
</div>
<div class="main">
  {conc_section}
  {buffett_section}
  {cape_section}
  {fed_section}
  {pe_ctx_section}
  {sector_section}
</div>
<div class="footer">
  Market Concentration &amp; Macro Valuation Dashboard &bull;
  Data: FRED, Shiller/Yale, TradingView &bull;
  Not investment advice.
</div>
{_HELP_MODAL}
<script>{_HELP_JS}</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Market Concentration & Macro Valuation Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-concentration", action="store_true", dest="no_conc",
                        help="Skip live S&P 500 concentration fetch (TradingView)")
    args = parser.parse_args()

    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))

    print("\n" + "=" * 58)
    print("  MARKET CONCENTRATION & MACRO VALUATION DASHBOARD")
    print("=" * 58)

    charts   = [None] * 6   # [conc, buffett, cape, fed, pe_context, sector_decomp]
    summary  = [(None, None, None), (None, None, None), (None, None, None), None]
    holdings = []
    cur_conc = None

    # ── 1. S&P 500 concentration ─────────────────────────────────────────────
    if not args.no_conc:
        print("\n[1/6] Fetching S&P 500 market concentration...")
        cur_conc, holdings = fetch_concentration()
        if cur_conc is not None:
            print(f"  Top-10 concentration: {cur_conc:.1f}%")
            for ticker, name, wt, mc in holdings[:10]:
                print(f"    {ticker:<6}  {wt:.2f}%  ${mc:,.0f}B  {name[:35]}")
        else:
            print("  Concentration data unavailable -- chart will use historical anchors only.")
    else:
        print("\n[1/6] Skipping live concentration fetch (--no-concentration).")
        cur_conc, holdings = None, []

    charts[0] = chart_concentration(CONCENTRATION_HIST, cur_conc, holdings)
    summary[0] = cur_conc

    # ── 2. Buffett Indicator ─────────────────────────────────────────────────
    print("\n[2/6] Fetching Buffett Indicator (FRED NCBEILQ027S / GDP)...")
    try:
        df_b = fetch_buffett()
        cur_b = float(df_b["ratio"].iloc[-1])
        mean_b = float(df_b["ratio"].mean())
        sd_b   = float(df_b["ratio"].std())
        print(f"  Current: {cur_b:.0f}%  |  Mean: {mean_b:.0f}%  |  +1 SD: {mean_b+sd_b:.0f}%")
        charts[1]  = chart_buffett(df_b)
        summary[1] = (cur_b, mean_b, sd_b)
    except Exception as e:
        print(f"  FAILED: {e}")
        summary[1] = (None, None, None)

    # ── 3. Shiller CAPE ──────────────────────────────────────────────────────
    print("\n[3/6] Fetching Shiller CAPE from Yale (ie_data.xls)...")
    df_shiller = None
    try:
        df_shiller = fetch_shiller()
        cur_c  = float(df_shiller["cape"].iloc[-1])
        mean_c = float(df_shiller["cape"].mean())
        sd_c   = float(df_shiller["cape"].std())
        cur_erp = float(df_shiller["erp"].iloc[-1]) if "erp" in df_shiller.columns else None
        if cur_erp is None and "gs10" in df_shiller.columns:
            df_shiller["erp"] = df_shiller["earnings_yield"] - df_shiller["gs10"]
            cur_erp = float(df_shiller["erp"].dropna().iloc[-1])
        print(f"  CAPE current: {cur_c:.1f}x  |  Mean: {mean_c:.1f}x  |  +1 SD: {mean_c+sd_c:.1f}x")
        print(f"  Equity Risk Premium: {cur_erp:+.2f}%")
        charts[2]  = chart_cape(df_shiller)
        summary[2] = (cur_c, mean_c, sd_c)
        summary[3] = cur_erp
    except Exception as e:
        print(f"  FAILED: {e}")
        summary[2] = (None, None, None)
        summary[3] = None

    # ── 4. Fed Model ─────────────────────────────────────────────────────────
    print("\n[4/6] Building Fed Model chart...")
    if df_shiller is not None:
        try:
            charts[3] = chart_fed_model(df_shiller)
        except Exception as e:
            print(f"  Fed Model chart failed: {e}")
    else:
        print("  Skipped (Shiller data unavailable).")

    # ── 5. Sector composition + live blended P/E ─────────────────────────────
    print("\n[5/6] Fetching sector composition & P/E breakdown...")
    df_sectors = None
    live_pe    = None
    try:
        df_sectors = fetch_sector_composition()
        if df_sectors is not None and len(df_sectors):
            live_pe = round(float(df_sectors["contribution"].sum()), 1)
            print(f"  Blended live P/E: {live_pe:.1f}x")
            for _, row in df_sectors.iterrows():
                print(f"    {row['sector']:<30}  {row['weight_pct']:5.1f}%  "
                      f"{row['pe_sector']:5.1f}x  -> {row['contribution']:.2f} pts")
        else:
            print("  Sector data unavailable.")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── 6. Valuation context charts ───────────────────────────────────────────
    print("\n[6/6] Building valuation context charts...")
    if df_shiller is not None:
        try:
            charts[4] = chart_pe_context(df_shiller, live_pe=live_pe)
            print("  CAPE vs Trailing P/E chart done.")
        except Exception as e:
            print(f"  CAPE context chart failed: {e}")
    if df_sectors is not None:
        try:
            charts[5] = chart_sector_decomp(df_sectors)
            print("  Sector P/E decomposition chart done.")
        except Exception as e:
            print(f"  Sector decomp chart failed: {e}")

    # ── Build HTML ────────────────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concentrationData")
    os.makedirs(out_dir, exist_ok=True)
    fname   = f"market_concentration_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fpath   = os.path.join(out_dir, fname)

    html = build_html(charts, summary, holdings, ts, suite_port=suite_port)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)

    kb = os.path.getsize(fpath) // 1024
    print(f"\n  HTML saved: {fpath}  ({kb} KB)")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        import platform, subprocess as _sp
        if platform.system() == "Darwin":
            _sp.Popen(["open", fpath])
        else:
            import webbrowser
            webbrowser.open(f"file://{fpath}")


if __name__ == "__main__":
    main()
