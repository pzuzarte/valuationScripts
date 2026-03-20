"""
regimeDetector.py
-----------------
Classifies the current macro market regime (Expansion, Late-Cycle,
Contraction, Recovery) using a Gaussian Mixture Model fitted on monthly
macro features sourced entirely from yfinance.

Usage:
    python regimeDetector.py [--lookback YEARS]

Output:
    19_regimeDetector/regimeData/regime_YYYYMMDD.html
"""

import sys
import os
import argparse
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── repo root on path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── output directory ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "regimeData")
os.makedirs(OUT_DIR, exist_ok=True)

# ── CSS / chart palette ────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1e2e"
BORDER  = "#252a3a"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
ACCENT  = "#6366f1"
GREEN   = "#10b981"
RED     = "#ef4444"
YELLOW  = "#f59e0b"
TEAL    = "#14b8a6"

REGIME_COLORS = {
    "Expansion":   GREEN,
    "Recovery":    TEAL,
    "Late-Cycle":  YELLOW,
    "Contraction": RED,
}

# ── WACC / growth recommendations ─────────────────────────────────────────────
REGIME_RECS = {
    "Expansion": {
        "wacc_adj": -0.5,
        "terminal_g_adj": +0.3,
        "note": "Risk appetite elevated — compress discount rate slightly; terminal growth can reflect above-trend GDP",
    },
    "Late-Cycle": {
        "wacc_adj": +0.5,
        "terminal_g_adj": -0.2,
        "note": "Cycle maturing — widen WACC for late-cycle risk; trim terminal growth assumption",
    },
    "Contraction": {
        "wacc_adj": +1.5,
        "terminal_g_adj": -0.5,
        "note": "Recession risk elevated — significantly widen WACC; use conservative terminal growth",
    },
    "Recovery": {
        "wacc_adj": 0.0,
        "terminal_g_adj": +0.1,
        "note": "Early recovery — WACC near baseline; growth can modestly exceed long-run GDP",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data(lookback_years: int) -> pd.DataFrame:
    """Download daily data for all tickers and return as DataFrame of daily closes."""
    end = datetime.today()
    start = end.replace(year=end.year - lookback_years)

    tickers = ["SPY", "^VIX", "^TNX", "^IRX", "HYG", "LQD"]
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Extract Close prices; handle both MultiIndex and flat column structures
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw[["Close"]].copy()

    closes.columns = [c.replace("^", "") for c in closes.columns]
    return closes


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to monthly close, construct features, forward-fill NaN.
    Returns a monthly DataFrame with columns:
        vix, yield_curve, credit_spread, spy_mom_3m, spy_mom_6m, rate_level
    Also preserves SPY monthly close for chart purposes.
    """
    # Monthly close (last business day of month)
    monthly = daily.resample("ME").last()

    # Forward-fill sparse series (e.g. VIX gaps on holidays)
    monthly = monthly.ffill()

    # ── individual series ──────────────────────────────────────────────────
    vix    = monthly["VIX"]
    tnx    = monthly["TNX"]          # 10Y yield (already in %)
    irx    = monthly["IRX"] / 10.0   # ^IRX quotes annualised 3M rate × 10
    hyg    = monthly["HYG"]
    lqd    = monthly["LQD"]
    spy    = monthly["SPY"]

    features = pd.DataFrame(index=monthly.index)
    features["spy"]          = spy
    features["vix"]          = vix
    features["rate_level"]   = tnx
    features["yield_curve"]  = tnx - irx                         # 10Y minus 3M (pp)
    features["credit_spread"] = np.log(lqd / hyg)               # proxy: wider = stress
    features["spy_mom_3m"]   = spy.pct_change(3)                 # 3-month return
    features["spy_mom_6m"]   = spy.pct_change(6)                 # 6-month return

    # Drop rows where core features are all NaN (first few months)
    feature_cols = ["vix", "yield_curve", "credit_spread",
                    "spy_mom_3m", "spy_mom_6m", "rate_level"]
    features = features.dropna(subset=feature_cols)
    features = features.ffill()

    return features, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# GMM FITTING & DETERMINISTIC LABELING
# ══════════════════════════════════════════════════════════════════════════════

def fit_gmm(X_scaled: np.ndarray) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42,
        n_init=10,
    )
    gmm.fit(X_scaled)
    return gmm


def label_regimes(gmm: GaussianMixture, scaler: StandardScaler,
                  feature_cols: list) -> dict:
    """
    Deterministically assign human-readable labels to GMM cluster IDs by
    inspecting unscaled centroids.

    Logic:
      - Sort the 4 centroids by VIX (ascending).
      - The two lowest-VIX clusters are split by spy_mom_3m:
          higher momentum → Expansion, lower → Recovery
      - The two highest-VIX clusters are split by yield_curve:
          more inverted (lower) → Contraction, flatter/more normal → Late-Cycle
    """
    # Inverse-transform centroids to original scale
    centroids_scaled = gmm.means_                          # shape (4, n_features)
    centroids = scaler.inverse_transform(centroids_scaled) # back to real units
    centroid_df = pd.DataFrame(centroids, columns=feature_cols)

    vix_order = centroid_df["vix"].argsort().values        # indices sorted low→high VIX

    low_vix_pair  = list(vix_order[:2])
    high_vix_pair = list(vix_order[2:])

    # Within low-VIX pair: higher spy_mom_3m → Expansion
    if centroid_df.loc[low_vix_pair[0], "spy_mom_3m"] >= \
       centroid_df.loc[low_vix_pair[1], "spy_mom_3m"]:
        expansion_id  = low_vix_pair[0]
        recovery_id   = low_vix_pair[1]
    else:
        expansion_id  = low_vix_pair[1]
        recovery_id   = low_vix_pair[0]

    # Within high-VIX pair: lower yield_curve (more inverted) → Contraction
    if centroid_df.loc[high_vix_pair[0], "yield_curve"] <= \
       centroid_df.loc[high_vix_pair[1], "yield_curve"]:
        contraction_id = high_vix_pair[0]
        latecycle_id   = high_vix_pair[1]
    else:
        contraction_id = high_vix_pair[1]
        latecycle_id   = high_vix_pair[0]

    id_to_label = {
        expansion_id:  "Expansion",
        recovery_id:   "Recovery",
        latecycle_id:  "Late-Cycle",
        contraction_id:"Contraction",
    }
    return id_to_label


# ══════════════════════════════════════════════════════════════════════════════
# PER-REGIME STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_stats(features: pd.DataFrame) -> pd.DataFrame:
    """
    For each regime label compute:
        avg_monthly_return, annual_vol, avg_duration_months, occurrences
    """
    features = features.copy()
    features["spy_ret"] = features["spy"].pct_change()

    rows = []
    for regime in ["Expansion", "Recovery", "Late-Cycle", "Contraction"]:
        mask = features["regime"] == regime
        subset = features[mask]
        if subset.empty:
            rows.append({
                "regime": regime,
                "avg_monthly_return": np.nan,
                "annual_vol": np.nan,
                "avg_duration": np.nan,
                "occurrences": 0,
            })
            continue

        avg_ret = subset["spy_ret"].mean() * 100       # percent
        ann_vol = subset["spy_ret"].std() * np.sqrt(12) * 100

        # Duration: count consecutive runs
        runs = []
        run_len = 1
        labels_list = features["regime"].tolist()
        for i in range(1, len(labels_list)):
            if labels_list[i] == labels_list[i - 1] == regime:
                run_len += 1
            elif labels_list[i - 1] == regime:
                runs.append(run_len)
                run_len = 1
        # close last run
        if labels_list[-1] == regime:
            runs.append(run_len)

        avg_dur = np.mean(runs) if runs else 1
        occurrences = len(runs)

        rows.append({
            "regime": regime,
            "avg_monthly_return": avg_ret,
            "annual_vol": ann_vol,
            "avg_duration": avg_dur,
            "occurrences": occurrences,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _sign(val: float) -> str:
    return f"+{val:.1f}" if val >= 0 else f"{val:.1f}"


def _arrow(series: pd.Series, col: str) -> str:
    """Return ↑ ↓ → based on 3-month change in a feature column."""
    if len(series) < 4:
        return "→"
    recent = series[col].iloc[-1]
    past   = series[col].iloc[-4]
    diff   = recent - past
    threshold = series[col].std() * 0.1
    if diff > threshold:
        return "↑"
    elif diff < -threshold:
        return "↓"
    return "→"


def _arrow_color(arrow: str) -> str:
    if arrow == "↑":
        return GREEN
    if arrow == "↓":
        return RED
    return MUTED


def generate_html(
    features: pd.DataFrame,
    feature_cols: list,
    current_regime: str,
    current_prob: float,
    regime_stats: pd.DataFrame,
    lookback_years: int,
) -> str:
    today_str = date.today().strftime("%B %d, %Y")
    today_file = date.today().strftime("%Y%m%d")

    recs = REGIME_RECS[current_regime]
    wacc_adj = recs["wacc_adj"]
    tg_adj   = recs["terminal_g_adj"]
    rec_note = recs["note"]
    reg_color = REGIME_COLORS[current_regime]

    # ── Chart.js data ──────────────────────────────────────────────────────
    chart_dates  = [d.strftime("%Y-%m") for d in features.index]
    spy_prices   = [round(float(v), 2) for v in features["spy"].values]
    pt_colors    = [REGIME_COLORS.get(r, MUTED) for r in features["regime"]]

    dates_js   = str(chart_dates).replace("'", '"')
    spy_js     = str(spy_prices)
    colors_js  = str(pt_colors).replace("'", '"')

    # ── Macro indicator current values ─────────────────────────────────────
    last = features.iloc[-1]
    indicators = [
        ("VIX",          f"{last['vix']:.1f}",          _arrow(features, "vix"),          ""),
        ("Yield Curve",  f"{last['yield_curve']:.2f}%",  _arrow(features, "yield_curve"),  " (10Y − 3M)"),
        ("Credit Spread",f"{last['credit_spread']:.3f}", _arrow(features, "credit_spread")," (log LQD/HYG)"),
        ("SPY 3M Return",f"{last['spy_mom_3m']*100:.1f}%",_arrow(features, "spy_mom_3m"), ""),
        ("Rate Level",   f"{last['rate_level']:.2f}%",   _arrow(features, "rate_level"),  " (10Y UST)"),
    ]

    # ── Regime stats rows ──────────────────────────────────────────────────
    stats_rows = ""
    for _, row in regime_stats.iterrows():
        rn = row["regime"]
        rc = REGIME_COLORS.get(rn, MUTED)
        ret_val = f"{row['avg_monthly_return']:.2f}%" if not np.isnan(row['avg_monthly_return']) else "—"
        vol_val = f"{row['annual_vol']:.1f}%"         if not np.isnan(row['annual_vol']) else "—"
        dur_val = f"{row['avg_duration']:.1f}"        if not np.isnan(row['avg_duration']) else "—"
        occ_val = str(int(row['occurrences']))
        ret_color = GREEN if (not np.isnan(row['avg_monthly_return']) and row['avg_monthly_return'] >= 0) else RED
        stats_rows += f"""
        <tr>
          <td><span class="badge" style="background:{rc}20;color:{rc};border:1px solid {rc}40">{rn}</span></td>
          <td style="color:{ret_color}">{ret_val}</td>
          <td>{vol_val}</td>
          <td>{dur_val}</td>
          <td>{occ_val}</td>
        </tr>"""

    # ── Macro indicator rows ───────────────────────────────────────────────
    ind_rows = ""
    for name, val, arrow, note in indicators:
        ac = _arrow_color(arrow)
        ind_rows += f"""
        <tr>
          <td>{name}<span style="color:{MUTED};font-size:0.78rem">{note}</span></td>
          <td style="font-weight:600">{val}</td>
          <td style="color:{ac};font-size:1.2rem">{arrow}</td>
        </tr>"""

    # ── Probability bar width ──────────────────────────────────────────────
    prob_pct = int(current_prob * 100)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Macro Regime Detector — {today_str}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: {BG};
      color: {TEXT};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      padding: 24px;
      min-height: 100vh;
    }}
    h1 {{ font-size: 1.6rem; font-weight: 700; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: {MUTED}; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .subtitle {{ color: {MUTED}; font-size: 0.85rem; margin-top: 4px; }}
    .panel {{
      background: {PANEL};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 22px 24px;
      margin-bottom: 20px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.82rem;
      letter-spacing: 0.04em;
    }}
    /* Regime card */
    .regime-name {{
      font-size: 2.4rem;
      font-weight: 800;
      color: {reg_color};
      letter-spacing: -0.02em;
      margin: 8px 0 4px;
    }}
    .prob-bar-wrap {{
      background: {BORDER};
      border-radius: 4px;
      height: 8px;
      width: 260px;
      margin: 10px 0 6px;
      overflow: hidden;
    }}
    .prob-bar {{
      background: {reg_color};
      height: 100%;
      width: {prob_pct}%;
      border-radius: 4px;
    }}
    .rec-box {{
      background: {BG};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 14px 18px;
      margin-top: 18px;
      display: flex;
      gap: 32px;
      flex-wrap: wrap;
    }}
    .rec-item {{ display: flex; flex-direction: column; gap: 4px; }}
    .rec-label {{ color: {MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .rec-val {{ font-size: 1.35rem; font-weight: 700; }}
    .rec-note {{ color: {MUTED}; font-size: 0.82rem; max-width: 520px; line-height: 1.5; }}
    /* Chart */
    .chart-wrap {{ position: relative; height: 320px; }}
    /* Tables */
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      text-align: left;
      color: {MUTED};
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 8px 12px;
      border-bottom: 1px solid {BORDER};
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid {BORDER}44;
      font-size: 0.88rem;
    }}
    tr:last-child td {{ border-bottom: none; }}
    /* Grid */
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    /* Methodology */
    .method-text {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.7; }}
    .method-text strong {{ color: {TEXT}; }}
    .header-row {{ display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 24px; }}
    .date-tag {{
      background: {PANEL};
      border: 1px solid {BORDER};
      border-radius: 6px;
      padding: 4px 12px;
      font-size: 0.82rem;
      color: {MUTED};
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <div class="header-row">
    <div>
      <h1>Macro Regime Detector</h1>
      <p class="subtitle">Gaussian Mixture Model · {lookback_years}-year lookback · Monthly features</p>
    </div>
    <span class="date-tag">{today_str}</span>
  </div>

  <!-- Current Regime Card -->
  <div class="panel">
    <h2>Current Regime</h2>
    <span class="badge" style="background:{reg_color}20;color:{reg_color};border:1px solid {reg_color}40;font-size:0.78rem">
      GMM Classification
    </span>
    <div class="regime-name">{current_regime}</div>
    <p style="color:{MUTED};font-size:0.9rem">{rec_note}</p>
    <div class="prob-bar-wrap"><div class="prob-bar"></div></div>
    <p style="color:{MUTED};font-size:0.8rem">Model confidence: <strong style="color:{TEXT}">{prob_pct}%</strong></p>

    <div class="rec-box">
      <div class="rec-item">
        <span class="rec-label">WACC Adjustment</span>
        <span class="rec-val" style="color:{'#ef4444' if wacc_adj > 0 else '#10b981' if wacc_adj < 0 else TEXT}">{_sign(wacc_adj)}%</span>
      </div>
      <div class="rec-item">
        <span class="rec-label">Terminal Growth Adj.</span>
        <span class="rec-val" style="color:{'#10b981' if tg_adj > 0 else '#ef4444' if tg_adj < 0 else TEXT}">{_sign(tg_adj)}%</span>
      </div>
      <div class="rec-item" style="flex:1">
        <span class="rec-label">DCF Guidance</span>
        <span class="rec-note" style="margin-top:6px">{rec_note}</span>
      </div>
    </div>
  </div>

  <!-- Timeline Chart -->
  <div class="panel">
    <h2>Historical Regime Classification</h2>
    <div class="chart-wrap">
      <canvas id="regimeChart"></canvas>
    </div>
  </div>

  <!-- Stats + Indicators Grid -->
  <div class="grid-2">

    <!-- Per-Regime Stats -->
    <div class="panel">
      <h2>Per-Regime SPY Statistics</h2>
      <table>
        <thead>
          <tr>
            <th>Regime</th>
            <th>Avg Monthly Ret</th>
            <th>Annual Vol</th>
            <th>Avg Duration</th>
            <th>Occurrences</th>
          </tr>
        </thead>
        <tbody>
          {stats_rows}
        </tbody>
      </table>
    </div>

    <!-- Macro Indicators -->
    <div class="panel">
      <h2>Current Macro Indicators</h2>
      <table>
        <thead>
          <tr><th>Indicator</th><th>Value</th><th>3M Trend</th></tr>
        </thead>
        <tbody>
          {ind_rows}
        </tbody>
      </table>
    </div>

  </div>

  <!-- Methodology -->
  <div class="panel">
    <h2>Methodology</h2>
    <p class="method-text">
      <strong>Model:</strong> A <strong>Gaussian Mixture Model (GMM)</strong> with 4 components is fitted on standardised
      monthly macro features covering {lookback_years} years of history.
      Features include VIX level, 10Y–3M yield curve spread, log(LQD/HYG) as a credit-spread proxy,
      SPY 3- and 6-month momentum, and the 10-year Treasury rate.
      Cluster labels are assigned deterministically by inspecting unscaled centroids:
      the two lowest-VIX clusters are split by momentum (Expansion vs Recovery),
      and the two highest-VIX clusters are split by yield-curve slope (Late-Cycle vs Contraction).
      <br><br>
      <strong>Data sources (all via yfinance):</strong>
      SPY (market proxy), ^VIX (CBOE Volatility Index), ^TNX (10Y Treasury yield),
      ^IRX (3-month T-Bill rate × 10), HYG (iShares High Yield Bond ETF), LQD (iShares IG Bond ETF).
      All data resampled to monthly frequency (last close of each month), forward-filled for sparse series.
      <br><br>
      <strong>WACC / terminal-growth recommendations</strong> are applied as basis-point overlays on top of
      the analyst's base-case DCF assumptions — they are directional guides, not absolute values.
    </p>
  </div>

  <script>
    const dates  = {dates_js};
    const spyPx  = {spy_js};
    const ptColors = {colors_js};

    const ctx = document.getElementById("regimeChart").getContext("2d");
    new Chart(ctx, {{
      type: "line",
      data: {{
        labels: dates,
        datasets: [{{
          label: "SPY Price",
          data: spyPx,
          borderColor: "{ACCENT}",
          borderWidth: 1.5,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: ptColors,
          pointBorderColor: ptColors,
          fill: false,
          tension: 0.25,
        }}]
      }},
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
              afterLabel: function(ctx) {{
                return "Regime: " + ["Expansion","Recovery","Late-Cycle","Contraction"][
                  ["{GREEN}","{TEAL}","{YELLOW}","{RED}"].indexOf(ptColors[ctx.dataIndex])
                ] || "";
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            ticks: {{ color: "{MUTED}", maxTicksLimit: 16, font: {{ size: 11 }} }},
            grid: {{ color: "{BORDER}" }}
          }},
          y: {{
            ticks: {{
              color: "{MUTED}",
              font: {{ size: 11 }},
              callback: v => "$" + v.toFixed(0)
            }},
            grid: {{ color: "{BORDER}" }}
          }}
        }}
      }}
    }});

    // Regime legend
    const legend = document.createElement("div");
    legend.style.cssText = "display:flex;gap:16px;margin-top:10px;flex-wrap:wrap";
    const regimes = [
      ["Expansion", "{GREEN}"],
      ["Recovery",  "{TEAL}"],
      ["Late-Cycle","{YELLOW}"],
      ["Contraction","{RED}"]
    ];
    regimes.forEach(([name, color]) => {{
      const dot = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{color}};margin-right:5px"></span>`;
      legend.innerHTML += `<span style="color:{TEXT};font-size:0.82rem">${{dot}}${{name}}</span>`;
    }});
    document.getElementById("regimeChart").parentElement.appendChild(legend);
  </script>

</body>
</html>
"""
    return html, today_file


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Macro Regime Detector — GMM-based market regime classification")
    parser.add_argument("--lookback", type=int, default=15, help="Years of history (default: 15)")
    args = parser.parse_args()

    lookback = args.lookback

    print(f"\n=== Regime Detector ===", flush=True)
    print(f"  Downloading macro data ({lookback}Y)...", flush=True)

    # ── Download ───────────────────────────────────────────────────────────
    daily = download_data(lookback)

    # ── Build features ─────────────────────────────────────────────────────
    features, feature_cols = build_features(daily)

    n_months = len(features)
    print(f"  Building feature matrix ({n_months} monthly observations)...", flush=True)

    if n_months < 24:
        print(f"\n  ERROR: Only {n_months} months of clean data available (need ≥ 24).")
        print(f"  Try reducing --lookback or check yfinance connectivity.")
        sys.exit(1)

    # ── Scale ──────────────────────────────────────────────────────────────
    X = features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit GMM ────────────────────────────────────────────────────────────
    print(f"  Fitting Gaussian Mixture Model (4 regimes)...", flush=True)
    gmm = fit_gmm(X_scaled)

    # ── Label regimes ──────────────────────────────────────────────────────
    id_to_label = label_regimes(gmm, scaler, feature_cols)

    raw_regimes = gmm.predict(X_scaled)
    features["regime"] = [id_to_label[r] for r in raw_regimes]

    # ── Current regime ─────────────────────────────────────────────────────
    x_last  = X_scaled[-1].reshape(1, -1)
    raw_cur  = gmm.predict(x_last)[0]
    probs    = gmm.predict_proba(x_last)[0]
    current_regime = id_to_label[raw_cur]
    current_prob   = probs[raw_cur]

    print(f"  Current regime: {current_regime} (p={current_prob:.1%})", flush=True)

    # ── Per-regime stats ───────────────────────────────────────────────────
    regime_stats = compute_regime_stats(features)

    # ── Generate HTML ──────────────────────────────────────────────────────
    print(f"  Generating report...", flush=True)
    html, today_file = generate_html(
        features=features,
        feature_cols=feature_cols,
        current_regime=current_regime,
        current_prob=current_prob,
        regime_stats=regime_stats,
        lookback_years=lookback,
    )

    out_path = os.path.join(OUT_DIR, f"regime_{today_file}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓  Report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
