"""
riskModel.py
Forward-looking portfolio risk dashboard.
Fits GARCH(1,1) models, computes VaR/CVaR, rolling beta, correlation matrix,
and renders a self-contained HTML report.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING: 'arch' package not installed. Falling back to historical vol for all tickers.", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEEPDIVE_CSV = "/Users/pzuzarte/GitHub/valuationScripts/deepDiveTickers/deepDiveTickers.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "riskData")

RISK_FREE_RATE = 0.04  # 4% annual

# CSS theme
BG      = "#0f1117"
PANEL   = "#1a1e2e"
BORDER  = "#252a3a"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
ACCENT  = "#6366f1"
GREEN   = "#10b981"
RED     = "#ef4444"
YELLOW  = "#f59e0b"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio Risk Model")
    parser.add_argument("--tickers",   type=str, default=None,
                        help="Comma-separated tickers (e.g. AAPL,MSFT,NVDA)")
    parser.add_argument("--weights",   type=str, default=None,
                        help="Comma-separated weights (must sum to 1). Equal-weight if omitted.")
    parser.add_argument("--benchmark", type=str, default="SPY",
                        help="Benchmark ticker (default: SPY)")
    parser.add_argument("--lookback",  type=int, default=504,
                        help="Days of price history (default: 504 = ~2 years)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tickers_from_csv(path: str):
    if not os.path.exists(path):
        return None, f"deepDiveTickers.csv not found at {path}"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Error reading {path}: {e}"
    if df.empty:
        return None, f"{path} is empty."
    # Use column named 'ticker' if present, else first column
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = df[col].dropna().str.strip().str.upper().tolist()
    tickers = [t for t in tickers if t]
    if not tickers:
        return None, f"No valid tickers found in {path}."
    return tickers, None


def download_prices(tickers: list, benchmark: str, lookback: int) -> pd.DataFrame:
    all_syms = list(dict.fromkeys(tickers + [benchmark]))  # deduplicated, order preserved
    end   = datetime.today()
    # Add buffer for weekends/holidays
    start = end - timedelta(days=int(lookback * 1.5))

    raw = yf.download(
        all_syms,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
        if len(all_syms) == 1:
            prices.columns = all_syms

    # Keep only the last `lookback` trading days
    prices = prices.dropna(how="all").tail(lookback)
    return prices


# ---------------------------------------------------------------------------
# GARCH fitting
# ---------------------------------------------------------------------------
def fit_garch(returns: pd.Series, horizon: int = 30):
    """
    Fit GARCH(1,1) on `returns` (daily, as decimals).
    Returns (garch_vol_annualized, fitted_conditional_vol_series).
    Falls back to historical vol on failure.
    """
    if not ARCH_AVAILABLE:
        ann_vol = returns.std() * math.sqrt(252)
        return ann_vol, None, False

    try:
        returns_pct = returns.dropna() * 100
        if len(returns_pct) < 60:
            raise ValueError("Insufficient data for GARCH")
        am  = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=horizon)
        garch_vol_30d = float(
            np.sqrt(forecast.variance.values[-1, :].mean())
        ) / 100 * math.sqrt(252)

        # Conditional vol history (annualized)
        cond_vol = pd.Series(
            np.sqrt(res.conditional_volatility.values) / 100 * math.sqrt(252),
            index=returns_pct.index,
            name=returns.name,
        )
        return garch_vol_30d, cond_vol, True
    except Exception:
        ann_vol = returns.std() * math.sqrt(252)
        return ann_vol, None, False


# ---------------------------------------------------------------------------
# Per-ticker metrics
# ---------------------------------------------------------------------------
def compute_var_cvar(returns: pd.Series, horizon: int = 30):
    r = returns.dropna()
    scale = math.sqrt(horizon)
    pct_5  = float(np.percentile(r, 5))
    pct_1  = float(np.percentile(r, 1))
    cvar_95 = float(r[r <= pct_5].mean())
    return pct_5 * scale, pct_1 * scale, cvar_95 * scale


def compute_rolling_beta(returns: pd.Series, bm_returns: pd.Series, window: int = 252):
    aligned = pd.concat([returns, bm_returns], axis=1).dropna()
    r  = aligned.iloc[:, 0]
    bm = aligned.iloc[:, 1]
    cov = r.rolling(window).cov(bm)
    var = bm.rolling(window).var()
    beta = (cov / var).iloc[-1]
    return float(beta) if not math.isnan(beta) else np.nan


def compute_sharpe(returns: pd.Series):
    r = returns.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std() * math.sqrt(252)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - RISK_FREE_RATE) / ann_vol


def compute_max_drawdown(returns: pd.Series):
    r = returns.dropna()
    cum = (1 + r).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()
    return float(drawdown)


def compute_rolling_realized_vol(returns: pd.Series, window: int = 21):
    """Rolling 21-day annualized realized volatility."""
    return returns.rolling(window).std() * math.sqrt(252)


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------
def compute_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    aligned = returns_df.dropna()
    port = aligned.dot(weights)
    port.name = "Portfolio"
    return port


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_f(v, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def risk_color(value, low_thresh, high_thresh, invert=False):
    """Return RED if high risk, GREEN if low risk."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MUTED
    if invert:
        # Higher = better (e.g. Sharpe)
        if value >= high_thresh:
            return GREEN
        elif value <= low_thresh:
            return RED
        return YELLOW
    else:
        # Higher = worse (e.g. vol, VaR)
        if value >= high_thresh:
            return RED
        elif value <= low_thresh:
            return GREEN
        return YELLOW


def corr_cell_color(corr_val):
    """Interpolate: -1=RED, 0=dark neutral, +1=BLUE."""
    if math.isnan(corr_val):
        return PANEL
    v = max(-1.0, min(1.0, corr_val))
    if v >= 0:
        # 0 -> neutral, 1 -> blue
        r = int(30  + (99  - 30)  * (1 - v))
        g = int(30  + (102 - 30)  * (1 - v))
        b = int(46  + (241 - 46)  * v)
    else:
        # 0 -> neutral, -1 -> red
        r = int(30  + (239 - 30)  * (-v))
        g = int(30  + (68  - 30)  * (1 + v))
        b = int(46  + (68  - 46)  * (1 + v))
    return f"rgb({r},{g},{b})"


def text_color_on_bg(hex_or_rgb: str):
    """Return black or white depending on background luminance."""
    if hex_or_rgb.startswith("rgb"):
        parts = hex_or_rgb[4:-1].split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        h = hex_or_rgb.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if lum > 128 else "#e2e8f0"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def build_html(
    tickers, weights, benchmark,
    per_ticker_metrics,      # dict: ticker -> dict of metrics
    portfolio_metrics,       # dict
    corr_matrix,             # pd.DataFrame
    garch_history,           # dict: ticker -> pd.Series (cond vol) or None
    realized_vol_history,    # dict: ticker -> pd.Series
    returns_df,              # pd.DataFrame of daily returns
    lookback,
    run_date,
):
    date_str = run_date.strftime("%Y-%m-%d")

    # -----------------------------------------------------------------------
    # Chart data — up to 4 tickers
    # -----------------------------------------------------------------------
    chart_tickers = tickers[:4]
    chart_colors  = [ACCENT, GREEN, YELLOW, RED]

    chart_datasets = []
    # Build unified date index from realized vol series
    all_dates = sorted(set(
        d for t in chart_tickers
        for d in realized_vol_history[t].dropna().index.strftime("%Y-%m-%d").tolist()
    ))

    for i, t in enumerate(chart_tickers):
        # Realized vol
        rv = realized_vol_history[t].dropna()
        rv_dict = {d.strftime("%Y-%m-%d"): round(float(v), 6) for d, v in rv.items()}
        rv_data = [rv_dict.get(d, None) for d in all_dates]

        chart_datasets.append({
            "label":           f"{t} Realized Vol",
            "data":            rv_data,
            "borderColor":     chart_colors[i],
            "backgroundColor": "transparent",
            "borderWidth":     1.5,
            "pointRadius":     0,
            "tension":         0.3,
        })

        # GARCH conditional vol
        gh = garch_history.get(t)
        if gh is not None:
            gh = gh.dropna()
            gh_dict = {d.strftime("%Y-%m-%d"): round(float(v), 6) for d, v in gh.items()}
            gh_data = [gh_dict.get(d, None) for d in all_dates]
            chart_datasets.append({
                "label":           f"{t} GARCH Vol",
                "data":            gh_data,
                "borderColor":     chart_colors[i],
                "backgroundColor": "transparent",
                "borderWidth":     2,
                "borderDash":      [5, 3],
                "pointRadius":     0,
                "tension":         0.3,
            })

    chart_labels_js  = json.dumps(all_dates)
    chart_datasets_js = json.dumps(chart_datasets)

    # -----------------------------------------------------------------------
    # Correlation heatmap table rows
    # -----------------------------------------------------------------------
    corr_tickers = list(corr_matrix.columns)
    corr_rows_html = ""
    for rt in corr_tickers:
        corr_rows_html += "        <tr>\n"
        corr_rows_html += f'          <td style="font-weight:600;color:{TEXT};padding:8px 10px;">{rt}</td>\n'
        for ct in corr_tickers:
            val = corr_matrix.loc[rt, ct]
            bg  = corr_cell_color(float(val))
            fg  = text_color_on_bg(bg)
            txt = "1.00" if rt == ct else f"{val:.2f}"
            corr_rows_html += (
                f'          <td style="background:{bg};color:{fg};'
                f'text-align:center;padding:8px 10px;font-size:0.82rem;">{txt}</td>\n'
            )
        corr_rows_html += "        </tr>\n"

    corr_header_cells = "".join(
        f'<th style="padding:8px 10px;color:{MUTED};font-weight:500;">{t}</th>'
        for t in corr_tickers
    )

    # -----------------------------------------------------------------------
    # Per-ticker table rows
    # -----------------------------------------------------------------------
    ticker_rows_html = ""
    for t in tickers:
        m = per_ticker_metrics[t]
        gv  = m.get("garch_vol")
        v95 = m.get("var_95")
        v99 = m.get("var_99")
        cv  = m.get("cvar_95")
        bt  = m.get("beta")
        sh  = m.get("sharpe")
        md  = m.get("max_dd")

        ticker_rows_html += "        <tr>\n"
        ticker_rows_html += f'          <td style="font-weight:700;color:{ACCENT};">{t}</td>\n'

        # GARCH vol
        c = risk_color(gv, 0.15, 0.40)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_pct(gv)}</td>\n'
        # VaR 95
        c = risk_color(abs(v95) if v95 else None, 0.05, 0.12)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_pct(v95)}</td>\n'
        # VaR 99
        c = risk_color(abs(v99) if v99 else None, 0.07, 0.18)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_pct(v99)}</td>\n'
        # CVaR 95
        c = risk_color(abs(cv) if cv else None, 0.06, 0.15)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_pct(cv)}</td>\n'
        # Beta
        c = risk_color(abs(bt) if bt is not None and not math.isnan(bt) else None, 0.5, 1.3)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_f(bt)}</td>\n'
        # Sharpe
        c = risk_color(sh if sh is not None and not math.isnan(sh) else None, 0.5, 1.5, invert=True)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_f(sh)}</td>\n'
        # Max DD
        c = risk_color(abs(md) if md is not None else None, 0.10, 0.30)
        ticker_rows_html += f'          <td style="color:{c};">{fmt_pct(md)}</td>\n'
        ticker_rows_html += "        </tr>\n"

    # -----------------------------------------------------------------------
    # Portfolio summary cards
    # -----------------------------------------------------------------------
    pv        = portfolio_metrics.get("garch_vol")
    p95       = portfolio_metrics.get("var_95")
    p99       = portfolio_metrics.get("var_99")
    pcv       = portfolio_metrics.get("cvar_95")
    div_ratio = portfolio_metrics.get("div_ratio")

    def _card(label, value, color=TEXT):
        return f"""
        <div style="background:{PANEL};border:1px solid {BORDER};border-radius:10px;
                    padding:20px 24px;min-width:160px;flex:1;">
          <div style="font-size:0.78rem;color:{MUTED};text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:8px;">{label}</div>
          <div style="font-size:1.65rem;font-weight:700;color:{color};">{value}</div>
        </div>"""

    pv_color  = risk_color(pv, 0.10, 0.25)
    v95_color = risk_color(abs(p95) if p95 else None, 0.04, 0.10)

    summary_cards = (
        _card("Portfolio GARCH Vol (Ann.)", fmt_pct(pv), pv_color)
        + _card("30d VaR 95%", fmt_pct(p95), v95_color)
        + _card("30d VaR 99%", fmt_pct(p99), risk_color(abs(p99) if p99 else None, 0.06, 0.15))
        + _card("CVaR 95%", fmt_pct(pcv), risk_color(abs(pcv) if pcv else None, 0.06, 0.15))
        + _card("Diversification Ratio", fmt_f(div_ratio), ACCENT)
    )

    # -----------------------------------------------------------------------
    # Risk interpretation
    # -----------------------------------------------------------------------
    port_hist_vol = portfolio_metrics.get("hist_vol", pv)
    med_vol       = portfolio_metrics.get("median_rolling_vol")

    if pv is None:
        regime_text = "Insufficient data to determine risk regime."
    elif med_vol and pv > med_vol * 1.25:
        regime_text = (
            f"The portfolio's current GARCH-estimated annualized volatility of "
            f"<strong>{fmt_pct(pv)}</strong> is meaningfully above the historical median "
            f"rolling volatility of <strong>{fmt_pct(med_vol)}</strong>. "
            "This suggests an <strong style='color:" + RED + "'>elevated risk regime</strong>. "
            "Consider reviewing position sizing, adding hedges, or increasing cash allocation."
        )
    elif med_vol and pv < med_vol * 0.80:
        regime_text = (
            f"The portfolio's current GARCH-estimated annualized volatility of "
            f"<strong>{fmt_pct(pv)}</strong> is below the historical median rolling volatility "
            f"of <strong>{fmt_pct(med_vol)}</strong>. "
            "This indicates a <strong style='color:" + GREEN + "'>low-volatility regime</strong>. "
            "Risk metrics look favorable, though calm regimes can precede sharp reversals."
        )
    else:
        regime_text = (
            f"The portfolio's current GARCH-estimated annualized volatility of "
            f"<strong>{fmt_pct(pv)}</strong> is near its historical median "
            f"({'N/A' if not med_vol else fmt_pct(med_vol)}). "
            "This suggests a <strong style='color:" + YELLOW + "'>normal risk regime</strong>. "
            "Monitor for shifts in macro conditions or concentration risk."
        )

    # -----------------------------------------------------------------------
    # Assemble HTML
    # -----------------------------------------------------------------------
    tickers_display = ", ".join(tickers)
    weights_display = ", ".join([f"{w:.1%}" for w in weights])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Portfolio Risk Model — {date_str}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: {BG}; color: {TEXT}; min-height: 100vh;
      padding: 32px 24px;
    }}
    h1 {{ font-size: 1.9rem; font-weight: 700; color: {TEXT}; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: {TEXT}; margin-bottom: 16px; }}
    .subtitle {{ color: {MUTED}; font-size: 0.88rem; margin-top: 6px; margin-bottom: 36px; }}
    .section {{ margin-bottom: 40px; }}
    .panel {{
      background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
      padding: 24px 28px;
    }}
    .cards-row {{
      display: flex; flex-wrap: wrap; gap: 14px; margin-top: 4px;
    }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.875rem; }}
    th {{
      background: {BG}; color: {MUTED}; font-weight: 500;
      text-align: left; padding: 10px 12px;
      border-bottom: 1px solid {BORDER};
      white-space: nowrap;
    }}
    td {{
      padding: 10px 12px; border-bottom: 1px solid {BORDER}; white-space: nowrap;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .chart-wrap {{ position: relative; height: 340px; }}
    .interp {{
      line-height: 1.7; font-size: 0.92rem; color: {TEXT};
    }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 16px;
      margin-bottom: 16px; font-size: 0.82rem; color: {MUTED};
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-dot {{ width: 12px; height: 3px; border-radius: 2px; }}
    .legend-dash {{
      width: 12px; height: 0; border-top: 2px dashed;
      display: inline-block;
    }}
  </style>
</head>
<body>

<!-- ===== HEADER ===== -->
<div class="section">
  <h1>Portfolio Risk Model</h1>
  <div class="subtitle">
    Tickers: {tickers_display} &nbsp;|&nbsp;
    Weights: {weights_display} &nbsp;|&nbsp;
    Benchmark: {benchmark} &nbsp;|&nbsp;
    Lookback: {lookback}d &nbsp;|&nbsp;
    Date: {date_str}
  </div>
</div>

<!-- ===== PORTFOLIO SUMMARY ===== -->
<div class="section">
  <h2>Portfolio Summary</h2>
  <div class="cards-row">
    {summary_cards}
  </div>
</div>

<!-- ===== PER-TICKER TABLE ===== -->
<div class="section">
  <h2>Per-Ticker Risk Metrics</h2>
  <div class="panel" style="padding:0;overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>GARCH Vol (Ann.)</th>
          <th>30d VaR 95%</th>
          <th>30d VaR 99%</th>
          <th>CVaR 95%</th>
          <th>Beta</th>
          <th>Sharpe</th>
          <th>Max DD</th>
        </tr>
      </thead>
      <tbody>
{ticker_rows_html}      </tbody>
    </table>
  </div>
</div>

<!-- ===== GARCH CHART ===== -->
<div class="section">
  <h2>GARCH Conditional vs. Realized Volatility (first {len(chart_tickers)} tickers)</h2>
  <div class="panel">
    <div class="legend">
      {''.join(
          f'<span class="legend-item">'
          f'<span class="legend-dot" style="background:{chart_colors[i]};"></span>'
          f'{chart_tickers[i]} Realized'
          f'</span>'
          f'<span class="legend-item">'
          f'<span class="legend-dash" style="border-color:{chart_colors[i]};"></span>'
          f'{chart_tickers[i]} GARCH'
          f'</span>'
          for i in range(len(chart_tickers))
      )}
    </div>
    <div class="chart-wrap">
      <canvas id="garchChart"></canvas>
    </div>
  </div>
</div>

<!-- ===== CORRELATION HEATMAP ===== -->
<div class="section">
  <h2>Correlation Matrix</h2>
  <div class="panel" style="padding:0;overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th style="background:{PANEL};"></th>
          {corr_header_cells}
        </tr>
      </thead>
      <tbody>
{corr_rows_html}      </tbody>
    </table>
  </div>
  <div style="font-size:0.78rem;color:{MUTED};margin-top:10px;">
    Color scale: <span style="color:{RED};">Red = -1.0 (inverse)</span> &nbsp;|&nbsp;
    <span style="color:{MUTED};">Dark = 0 (uncorrelated)</span> &nbsp;|&nbsp;
    <span style="color:#6366f1;">Blue = +1.0 (perfect correlation)</span>
  </div>
</div>

<!-- ===== RISK INTERPRETATION ===== -->
<div class="section">
  <h2>Risk Interpretation</h2>
  <div class="panel">
    <p class="interp">{regime_text}</p>
    <p class="interp" style="margin-top:12px;color:{MUTED};font-size:0.83rem;">
      VaR figures are historical simulation scaled to 30 days (×√30). GARCH volatility is the
      30-horizon average conditional variance from a GARCH(1,1) model, annualized.
      Sharpe uses a 4% risk-free rate. Beta is estimated from a 252-day rolling window vs {benchmark}.
    </p>
  </div>
</div>

<script>
(function() {{
  const labels   = {chart_labels_js};
  const datasets = {chart_datasets_js};

  // Convert dashes to actual borderDash arrays for Chart.js
  datasets.forEach(ds => {{
    if (ds.borderDash) {{
      ds.borderDash = ds.borderDash;
    }}
  }});

  const ctx = document.getElementById("garchChart").getContext("2d");
  new Chart(ctx, {{
    type: "line",
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: "{PANEL}",
          borderColor: "{BORDER}",
          borderWidth: 1,
          titleColor: "{TEXT}",
          bodyColor: "{MUTED}",
          callbacks: {{
            label: ctx => {{
              const v = ctx.parsed.y;
              return v != null ? ` ${{ctx.dataset.label}}: ${{(v * 100).toFixed(1)}}%` : "";
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{
            color: "{MUTED}",
            maxTicksLimit: 10,
            maxRotation: 0,
          }},
          grid: {{ color: "{BORDER}" }},
        }},
        y: {{
          ticks: {{
            color: "{MUTED}",
            callback: v => (v * 100).toFixed(0) + "%"
          }},
          grid: {{ color: "{BORDER}" }},
          title: {{
            display: true,
            text: "Annualized Volatility",
            color: "{MUTED}",
            font: {{ size: 11 }}
          }}
        }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"\n=== Portfolio Risk Model ===", flush=True)

    # --- Resolve tickers ---
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers, err = load_tickers_from_csv(DEEPDIVE_CSV)
        if err:
            print(f"\nERROR: {err}")
            print("Provide tickers via --tickers or ensure deepDiveTickers.csv exists and is populated.")
            sys.exit(1)

    if len(tickers) < 2:
        print("\nERROR: At least 2 tickers are required for a correlation matrix.")
        print(f"  Found: {tickers}")
        sys.exit(1)

    print(f"  Tickers: {', '.join(tickers)}", flush=True)

    # --- Resolve weights ---
    if args.weights:
        raw_w = [float(w.strip()) for w in args.weights.split(",")]
        if len(raw_w) != len(tickers):
            print(f"\nERROR: {len(raw_w)} weights supplied but {len(tickers)} tickers.")
            sys.exit(1)
        s = sum(raw_w)
        weights = np.array([w / s for w in raw_w])  # normalise just in case
    else:
        weights = np.array([1.0 / len(tickers)] * len(tickers))

    benchmark = args.benchmark.upper()
    lookback  = args.lookback

    # --- Download prices ---
    print(f"  Downloading price history ({lookback}d)...", flush=True)
    prices = download_prices(tickers, benchmark, lookback)

    # Validate all tickers present
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"\nWARNING: Could not retrieve data for: {', '.join(missing)}")
        tickers  = [t for t in tickers if t not in missing]
        weights  = weights[[i for i, t in enumerate(tickers + missing) if t not in missing]]
        weights /= weights.sum()
        if len(tickers) < 2:
            print("ERROR: Fewer than 2 valid tickers remain. Exiting.")
            sys.exit(1)

    if benchmark not in prices.columns:
        print(f"\nERROR: Could not retrieve benchmark data for {benchmark}.")
        sys.exit(1)

    # Compute returns
    all_returns = prices.pct_change().dropna(how="all")
    bm_returns  = all_returns[benchmark]
    returns_df  = all_returns[tickers]

    # --- Fit GARCH + per-ticker metrics ---
    print(f"  Fitting GARCH(1,1) models...", flush=True)

    per_ticker_metrics  = {}
    garch_history       = {}
    realized_vol_history = {}

    for t in tickers:
        r = returns_df[t].dropna()

        garch_vol, cond_vol, garch_ok = fit_garch(r)
        if not garch_ok:
            print(f"    {t}: GARCH did not converge — using historical vol", flush=True)
        else:
            print(f"    {t}: GARCH vol {garch_vol:.1%} ann.", flush=True)

        var_95, var_99, cvar_95 = compute_var_cvar(r)
        beta     = compute_rolling_beta(r, bm_returns)
        sharpe   = compute_sharpe(r)
        max_dd   = compute_max_drawdown(r)

        per_ticker_metrics[t] = {
            "garch_vol": garch_vol,
            "garch_ok":  garch_ok,
            "var_95":    var_95,
            "var_99":    var_99,
            "cvar_95":   cvar_95,
            "beta":      beta,
            "sharpe":    sharpe,
            "max_dd":    max_dd,
        }
        garch_history[t]        = cond_vol
        realized_vol_history[t] = compute_rolling_realized_vol(r)

    # --- Portfolio-level metrics ---
    print(f"  Computing portfolio metrics...", flush=True)

    port_returns = compute_portfolio_returns(returns_df, weights)

    p_garch_vol, _, p_garch_ok = fit_garch(port_returns)
    p_var_95, p_var_99, p_cvar_95 = compute_var_cvar(port_returns)

    # Diversification ratio: wtd avg individual ann vol / portfolio ann vol
    indiv_vols  = np.array([returns_df[t].std() * math.sqrt(252) for t in tickers])
    wtd_avg_vol = float(np.dot(weights, indiv_vols))
    port_ann_vol = port_returns.std() * math.sqrt(252)
    div_ratio   = wtd_avg_vol / port_ann_vol if port_ann_vol > 0 else np.nan

    # Rolling realized vol for regime detection
    port_rolling_vol = compute_rolling_realized_vol(port_returns)
    median_rolling   = float(port_rolling_vol.dropna().median())

    portfolio_metrics = {
        "garch_vol":        p_garch_vol,
        "var_95":           p_var_95,
        "var_99":           p_var_99,
        "cvar_95":          p_cvar_95,
        "div_ratio":        div_ratio,
        "hist_vol":         port_ann_vol,
        "median_rolling_vol": median_rolling,
    }

    print(
        f"  Portfolio GARCH vol: {p_garch_vol:.1%}  VaR95: {p_var_95:.1%}",
        flush=True,
    )

    # --- Correlation matrix ---
    corr_matrix = returns_df.corr()

    # --- Generate HTML ---
    print(f"  Generating report...", flush=True)

    run_date = datetime.today()
    os.makedirs(OUT_DIR, exist_ok=True)
    fname    = f"risk_model_{run_date.strftime('%Y%m%d')}.html"
    out_path = os.path.join(OUT_DIR, fname)

    html = build_html(
        tickers             = tickers,
        weights             = weights,
        benchmark           = benchmark,
        per_ticker_metrics  = per_ticker_metrics,
        portfolio_metrics   = portfolio_metrics,
        corr_matrix         = corr_matrix,
        garch_history       = garch_history,
        realized_vol_history= realized_vol_history,
        returns_df          = returns_df,
        lookback            = lookback,
        run_date            = run_date,
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"  ✓  Report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
