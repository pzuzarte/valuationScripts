#!/usr/bin/env python3
"""
11_priceForecast/priceForecast.py

Price / return forecasting using:
  · ARIMA  — auto-order selection via pmdarima (auto_arima)
  · ETS    — additive damped-trend Holt-Winters (statsmodels)

Walk-forward backtest compares both models against a naive random-walk baseline.

Usage:
  python priceForecast.py --ticker AAPL --horizon 30 --model both --period 5y

Output: forecastData/YYYY_MM_DD_TICKER_forecast.html  (auto-opens in browser)
"""

import argparse
import datetime
import json
import os
import sys
import warnings
import webbrowser

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance required.  pip install yfinance")

try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False
    print("[warn] pmdarima not found — ARIMA disabled.  pip install pmdarima")

try:
    from arch import arch_model as _arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("[warn] arch not found — GARCH disabled.  pip install arch")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as _acf

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "forecastData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BACKTEST_DAYS   = 252          # 1 year walk-forward window
MIN_TRAIN_DAYS  = 252          # minimum history required before backtesting
CI_95_Z         = 1.960
CI_80_Z         = 1.282
PLOTLY_CDN      = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# ── Dark-theme colours (matches the rest of the suite) ─────────────────────────
DARK_BG     = "#161b2e"
CARD_BG     = "#1e2538"
BORDER      = "#252a3a"
TEXT_MAIN   = "#e8eaf0"
TEXT_SUB    = "#6b7194"
ACCENT_BLUE = "#4f8ef7"
ACCENT_GRN  = "#00c896"
ACCENT_RED  = "#e05c5c"
ACCENT_AMB  = "#f0a500"
ACCENT_PRP  = "#a78bfa"


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
def fetch_prices(ticker: str, period: str) -> pd.Series:
    print(f"  Fetching {ticker} history ({period})…", flush=True)
    h = yf.Ticker(ticker).history(period=period)
    if h.empty:
        sys.exit(f"ERROR: no price data returned for {ticker!r}")
    prices = h["Close"].dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    print(f"  {len(prices)} trading days  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})", flush=True)
    return prices


def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":     info.get("longName", ticker),
            "sector":   info.get("sector",   "—"),
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "—", "currency": "USD"}


# ══════════════════════════════════════════════════════════════════════════════
# Log-return helpers
# ══════════════════════════════════════════════════════════════════════════════
def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def returns_to_prices(start_price: float, ret_array: np.ndarray) -> np.ndarray:
    """Cumulative back-transform: start_price × exp(cumsum(returns))."""
    return start_price * np.exp(np.cumsum(ret_array))


def price_ci(start_price: float, cum_returns: np.ndarray,
             resid_std: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat-σ uncertainty cone (σ × √h random-walk approximation).
    Used as fallback when GARCH is unavailable.
    """
    steps = np.arange(1, len(cum_returns) + 1)
    upper_log = cum_returns + z * resid_std * np.sqrt(steps)
    lower_log = cum_returns - z * resid_std * np.sqrt(steps)
    return (start_price * np.exp(lower_log),
            start_price * np.exp(upper_log))


def price_ci_garch(start_price: float, cum_returns: np.ndarray,
                   garch_var_path: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    GARCH-based uncertainty cone.
    garch_var_path: h-step conditional variances in log-return² space.
    Cumulative variance at step h = Σ var[0..h-1] (integrated variance path).
    """
    h = len(cum_returns)
    cum_var   = np.cumsum(garch_var_path[:h])
    upper_log = cum_returns + z * np.sqrt(cum_var)
    lower_log = cum_returns - z * np.sqrt(cum_var)
    return (start_price * np.exp(lower_log),
            start_price * np.exp(upper_log))


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════════════════════════════════
def fit_arima(ret: pd.Series) -> tuple:
    """
    Fit auto_arima on log-returns.  Returns (order, resid_std, model_fit).
    """
    print("  Fitting ARIMA (auto_arima)…", flush=True)
    am = auto_arima(
        ret,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        max_d=2,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    order  = am.order
    resid  = am.resid()
    print(f"  ARIMA{order}  AIC={am.aic():.1f}  BIC={am.bic():.1f}", flush=True)
    return order, float(np.std(resid)), am


def arima_forecast(ret: pd.Series, order: tuple,
                   horizon: int) -> tuple[np.ndarray, float]:
    """Fit fixed-order ARIMA on ret, return (forecast_returns, resid_std)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(ret, order=order).fit()
    fc  = fit.forecast(steps=horizon).values
    std = float(np.std(fit.resid))
    return fc, std


# ══════════════════════════════════════════════════════════════════════════════
# ETS (Holt-Winters)
# ══════════════════════════════════════════════════════════════════════════════
def fit_ets(log_px: pd.Series) -> object:
    """Additive damped-trend ETS on log prices (no seasonality)."""
    print("  Fitting ETS (additive damped trend)…", flush=True)
    model = ExponentialSmoothing(
        log_px,
        trend="add",
        damped_trend=True,
        seasonal=None,
    )
    fit = model.fit(optimized=True, use_brute=False)
    print(f"  ETS  AIC={fit.aic:.1f}", flush=True)
    return fit


def ets_forecast(log_px: pd.Series, horizon: int) -> tuple[np.ndarray, float]:
    """Fit fresh ETS, return (forecast_prices, resid_std)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(log_px, trend="add",
                                     damped_trend=True, seasonal=None)
        fit   = model.fit(optimized=True, use_brute=False)
    fc_log = fit.forecast(horizon).values
    std    = float(np.std(fit.resid))
    return np.exp(fc_log), std          # back to price space


# ══════════════════════════════════════════════════════════════════════════════
# GARCH(1,1) — volatility model
# ══════════════════════════════════════════════════════════════════════════════
def fit_garch(ret: pd.Series):
    """
    Fit GARCH(1,1) with Normal innovations on log returns.
    Input is scaled ×100 for numerical stability; variance is unscaled on output.
    Returns the fitted arch ModelResult.
    """
    model = _arch_model(ret * 100, vol="Garch", p=1, q=1,
                        dist="normal", rescale=False)
    fit = model.fit(disp="off", show_warning=False)
    omega = fit.params.get("omega", 0.0)
    alpha = fit.params.get("alpha[1]", fit.params.get("alpha", 0.0))
    beta  = fit.params.get("beta[1]",  fit.params.get("beta",  0.0))
    persistence = alpha + beta
    lr_vol = np.sqrt(omega / max(1 - persistence, 1e-6)) * np.sqrt(252)
    print(f"  GARCH(1,1)  ω={omega:.4f}  α={alpha:.4f}  β={beta:.4f}  "
          f"persistence={persistence:.4f}  LR ann.vol={lr_vol:.1f}%", flush=True)
    return fit


def garch_var_path(garch_fit, horizon: int) -> np.ndarray:
    """
    h-step ahead conditional variances in log-return² space (not % space).
    arch fit used ret×100, so variance is in (100×ret)² → divide by 10 000.
    Returns shape (horizon,).
    """
    fc  = garch_fit.forecast(horizon=horizon, reindex=False)
    var = fc.variance.values[-1]   # (horizon,), in (100×ret)² units
    return var / 10_000            # → log-return² units


def garch_hist_ann_vol(garch_fit) -> np.ndarray:
    """
    Historical conditional volatility from the fitted model, annualised in %.
    conditional_volatility is daily vol in (100×ret) units → ×√252 = ann %.
    """
    return (garch_fit.conditional_volatility * np.sqrt(252)).values


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward back-test
# ══════════════════════════════════════════════════════════════════════════════
def backtest(prices: pd.Series, horizon: int,
             arima_order: tuple | None,
             do_arima: bool, do_ets: bool) -> dict:
    """
    Walk-forward backtest over the last BACKTEST_DAYS trading days.
    Each step: train on history up to t, predict next `horizon` days,
    step forward `horizon` days, repeat.

    Returns dict with keys: dates, actual, arima_preds, ets_preds,
    naive_preds, arima_metrics, ets_metrics, naive_metrics.
    """
    n = len(prices)
    if n < MIN_TRAIN_DAYS + horizon:
        return {}

    # Slice so the test window is exactly BACKTEST_DAYS (or fewer if not enough data)
    bt_days   = min(BACKTEST_DAYS, n - MIN_TRAIN_DAYS)
    test_start = n - bt_days
    n_steps   = max(1, bt_days // horizon)

    all_actual     = []
    all_arima_pred = []
    all_ets_pred   = []
    all_naive_pred = []
    all_dates      = []

    print(f"  Walk-forward backtest: {n_steps} step(s) × {horizon} days…",
          flush=True)

    for step in range(n_steps):
        t0 = test_start + step * horizon
        t1 = t0 + horizon
        if t1 > n:
            break

        train_px  = prices.iloc[:t0]
        actual_px = prices.iloc[t0:t1].values
        last_px   = float(train_px.iloc[-1])

        # Dates for this segment
        all_dates.extend([str(d.date()) for d in prices.index[t0:t1]])
        all_actual.extend(actual_px.tolist())

        # Naive: flat (random walk — last price repeated)
        all_naive_pred.extend([last_px] * horizon)

        # ARIMA
        if do_arima and arima_order is not None:
            try:
                ret_train = log_returns(train_px)
                fc_ret, _  = arima_forecast(ret_train, arima_order, horizon)
                fc_px      = returns_to_prices(last_px, fc_ret)
                all_arima_pred.extend(fc_px.tolist())
            except Exception:
                all_arima_pred.extend([np.nan] * horizon)
        else:
            all_arima_pred.extend([np.nan] * horizon)

        # ETS
        if do_ets:
            try:
                log_train = np.log(train_px)
                fc_px_ets, _ = ets_forecast(log_train, horizon)
                all_ets_pred.extend(fc_px_ets.tolist())
            except Exception:
                all_ets_pred.extend([np.nan] * horizon)
        else:
            all_ets_pred.extend([np.nan] * horizon)

    actual_arr = np.array(all_actual)

    def metrics(pred_list):
        pred = np.array(pred_list)
        mask = ~np.isnan(pred)
        if mask.sum() == 0:
            return {}
        a, p = actual_arr[mask], pred[mask]
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        mae  = float(np.mean(np.abs(a - p)))
        mape = float(np.mean(np.abs((a - p) / a)) * 100)
        # Directional accuracy over horizon steps
        a_dir = np.sign(np.diff(a))
        p_dir = np.sign(np.diff(p))
        dir_mask = a_dir != 0
        if dir_mask.sum() > 0:
            dacc = float((a_dir[dir_mask] == p_dir[dir_mask]).mean() * 100)
        else:
            dacc = float("nan")
        return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": dacc}

    return {
        "dates":          all_dates,
        "actual":         all_actual,
        "arima_preds":    all_arima_pred,
        "ets_preds":      all_ets_pred,
        "naive_preds":    all_naive_pred,
        "arima_metrics":  metrics(all_arima_pred) if do_arima else {},
        "ets_metrics":    metrics(all_ets_pred)   if do_ets   else {},
        "naive_metrics":  metrics(all_naive_pred),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTML generation
# ══════════════════════════════════════════════════════════════════════════════
def _js_list(arr, ndigits=4):
    """Serialize a list/array to compact JSON, replacing NaN with null."""
    def _fix(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(float(v), ndigits)
    return json.dumps([_fix(v) for v in arr])


def _color_delta(v):
    if v is None:
        return TEXT_SUB
    return ACCENT_GRN if v >= 0 else ACCENT_RED


def _fmt_pct(v, decimals=2):
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def _metric_row(label, val, baseline, lower_is_better=True):
    """Coloured table row: green if model beats baseline, red otherwise."""
    if not val or not baseline:
        return (f'<tr><td>{label}</td><td>—</td><td>—</td>'
                f'<td style="color:{TEXT_SUB}">—</td></tr>')
    beats  = val < baseline if lower_is_better else val > baseline
    delta  = val - baseline
    clr    = ACCENT_GRN if beats else ACCENT_RED
    sign   = "+" if delta >= 0 else ""
    return (f'<tr><td>{label}</td>'
            f'<td>{val:.4f}</td>'
            f'<td>{baseline:.4f}</td>'
            f'<td style="color:{clr}">{sign}{delta:.4f}</td></tr>')


def build_html(
    ticker: str,
    info: dict,
    prices: pd.Series,
    horizon: int,
    model_arg: str,
    arima_fit,
    arima_order,
    arima_resid_std: float,
    ets_fit,
    ets_resid_std: float,
    bt: dict,
    do_arima: bool,
    do_ets: bool,
    garch_fit=None,
    garch_var=None,        # shape (horizon,), log-return² units
) -> str:

    today       = datetime.date.today()
    last_price  = float(prices.iloc[-1])
    currency    = info.get("currency", "USD")
    last_date   = prices.index[-1].date()

    # ── Forecast data ──────────────────────────────────────────────────────────
    future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1),
                                  periods=horizon)
    future_strs  = [str(d.date()) for d in future_dates]

    # Decide CI method
    use_garch_ci = (garch_fit is not None and garch_var is not None)

    # ARIMA forecast
    arima_fc_px = arima_fc_lo80 = arima_fc_hi80 = \
    arima_fc_lo95 = arima_fc_hi95 = []
    if do_arima and arima_fit is not None:
        fc_ret   = arima_fit.predict(n_periods=horizon)
        cum_ret  = np.cumsum(fc_ret)
        arima_fc_px = returns_to_prices(last_price, fc_ret).tolist()
        if use_garch_ci:
            lo80, hi80 = price_ci_garch(last_price, cum_ret, garch_var, CI_80_Z)
            lo95, hi95 = price_ci_garch(last_price, cum_ret, garch_var, CI_95_Z)
        else:
            lo80, hi80 = price_ci(last_price, cum_ret, arima_resid_std, CI_80_Z)
            lo95, hi95 = price_ci(last_price, cum_ret, arima_resid_std, CI_95_Z)
        arima_fc_lo80 = lo80.tolist();  arima_fc_hi80 = hi80.tolist()
        arima_fc_lo95 = lo95.tolist();  arima_fc_hi95 = hi95.tolist()

    # ETS forecast
    ets_fc_px = ets_fc_lo80 = ets_fc_hi80 = \
    ets_fc_lo95 = ets_fc_hi95 = []
    if do_ets and ets_fit is not None:
        fc_log    = ets_fit.forecast(horizon).values
        ets_fc_px = np.exp(fc_log).tolist()
        if use_garch_ci:
            # Use GARCH variance path for ETS CI too — same vol regime applies
            cum_ret_ets = fc_log - np.log(last_price)   # log-return equivalent
            lo80, hi80 = price_ci_garch(last_price, cum_ret_ets, garch_var, CI_80_Z)
            lo95, hi95 = price_ci_garch(last_price, cum_ret_ets, garch_var, CI_95_Z)
        else:
            steps    = np.arange(1, horizon + 1)
            lo80 = np.exp(fc_log - CI_80_Z * ets_resid_std * np.sqrt(steps))
            hi80 = np.exp(fc_log + CI_80_Z * ets_resid_std * np.sqrt(steps))
            lo95 = np.exp(fc_log - CI_95_Z * ets_resid_std * np.sqrt(steps))
            hi95 = np.exp(fc_log + CI_95_Z * ets_resid_std * np.sqrt(steps))
        ets_fc_lo80 = lo80.tolist();  ets_fc_hi80 = hi80.tolist()
        ets_fc_lo95 = lo95.tolist();  ets_fc_hi95 = hi95.tolist()

    # Naive forecast (flat)
    naive_fc_px = [last_price] * horizon

    # Historical window for chart (last 6 months ≈ 126 days)
    hist_window  = min(126, len(prices))
    hist_prices  = prices.iloc[-hist_window:]
    hist_dates   = [str(d.date()) for d in hist_prices.index]
    hist_vals    = hist_prices.values.tolist()

    # ── ARIMA residual ACF ─────────────────────────────────────────────────────
    acf_vals = []
    if do_arima and arima_fit is not None:
        try:
            raw_acf = _acf(arima_fit.resid(), nlags=20, fft=True)[1:]  # skip lag 0
            acf_vals = [round(float(v), 4) for v in raw_acf]
        except Exception:
            pass
    acf_lags = list(range(1, len(acf_vals) + 1))

    # ── Log-return distribution ────────────────────────────────────────────────
    ret_all_vals = log_returns(prices).values
    ret_hist_counts, ret_hist_edges = np.histogram(ret_all_vals, bins=60)
    ret_hist_centers = [(ret_hist_edges[i] + ret_hist_edges[i+1]) / 2
                        for i in range(len(ret_hist_edges)-1)]

    # ── Metrics ────────────────────────────────────────────────────────────────
    arima_m = bt.get("arima_metrics", {})
    ets_m   = bt.get("ets_metrics",   {})
    naive_m = bt.get("naive_metrics", {})

    # ── End-of-horizon price summary ──────────────────────────────────────────
    def _summary(fc_px_list):
        if not fc_px_list:
            return None, None
        end_px  = fc_px_list[-1]
        pct_chg = (end_px / last_price - 1) * 100
        return end_px, pct_chg

    arima_end, arima_chg = _summary(arima_fc_px)
    ets_end,   ets_chg   = _summary(ets_fc_px)

    # ── ARIMA model text ───────────────────────────────────────────────────────
    arima_summary_html = ""
    if do_arima and arima_fit is not None:
        p, d, q = arima_order
        arima_summary_html = f"""
        <div class="card" style="margin-top:16px">
          <h3>ARIMA Model Summary</h3>
          <table class="summary-table">
            <tr><td>Order</td><td>ARIMA({p},{d},{q})</td></tr>
            <tr><td>AIC</td><td>{arima_fit.aic():.2f}</td></tr>
            <tr><td>BIC</td><td>{arima_fit.bic():.2f}</td></tr>
            <tr><td>Residual σ</td><td>{arima_resid_std:.6f} (log-return units)</td></tr>
            <tr><td>Training obs</td><td>{len(log_returns(prices))}</td></tr>
          </table>
          <p class="note" style="margin-top:10px">
            ARIMA fitted on log-returns (stationary series). Forecast is back-transformed
            to price space via cumulative exponentiation. Confidence cone assumes variance
            grows as σ² × h (random-walk / integrated-ARIMA approximation).
          </p>
        </div>"""

    ets_summary_html = ""
    if do_ets and ets_fit is not None:
        ets_summary_html = f"""
        <div class="card" style="margin-top:16px">
          <h3>ETS Model Summary</h3>
          <table class="summary-table">
            <tr><td>Type</td><td>Additive damped trend (no seasonality)</td></tr>
            <tr><td>AIC</td><td>{ets_fit.aic:.2f}</td></tr>
            <tr><td>α (level)</td><td>{ets_fit.params.get('smoothing_level', float('nan')):.4f}</td></tr>
            <tr><td>β (trend)</td><td>{ets_fit.params.get('smoothing_trend', float('nan')):.4f}</td></tr>
            <tr><td>φ (damping)</td><td>{ets_fit.params.get('damping_trend', float('nan')):.4f}</td></tr>
            <tr><td>Residual σ</td><td>{ets_resid_std:.6f} (log-price units)</td></tr>
          </table>
          <p class="note" style="margin-top:10px">
            ETS fitted on log prices to capture trend in price levels directly.
          </p>
        </div>"""

    # ── Metrics table rows ─────────────────────────────────────────────────────
    def _metric_td(m, key, baseline_m, lower_is_better=True):
        v = m.get(key)
        b = baseline_m.get(key)
        if v is None or b is None:
            return "<td>—</td><td>—</td>"
        beats = v < b if lower_is_better else v > b
        clr   = ACCENT_GRN if beats else ACCENT_RED
        delta = v - b
        sign  = "+" if delta >= 0 else ""
        return f'<td>{v:.4f}</td><td style="color:{clr}">{sign}{delta:.4f} vs naive</td>'

    def _dir_td(m, baseline_m):
        v = m.get("dir_acc")
        b = baseline_m.get("dir_acc")
        if v is None:
            return "<td>—</td><td>—</td>"
        b_str = f"{b:.1f}%" if b is not None else "—"
        clr   = ACCENT_GRN if (b is not None and v > b) else ACCENT_RED
        delta = (v - b) if b is not None else 0
        sign  = "+" if delta >= 0 else ""
        return (f'<td>{v:.1f}%</td>'
                f'<td style="color:{clr}">{sign}{delta:.1f}pp vs naive</td>')

    metrics_html = ""
    if bt:
        rows = ""
        for label, m in [("ARIMA", arima_m), ("ETS", ets_m)]:
            if not m:
                continue
            rows += f"""
            <tr>
              <td><b>{label}</b></td>
              {_metric_td(m, 'rmse', naive_m)}
              {_metric_td(m, 'mae',  naive_m)}
              {_metric_td(m, 'mape', naive_m)}
              {_dir_td(m, naive_m)}
            </tr>"""
        rows += f"""
            <tr style="color:{TEXT_SUB}">
              <td>Naive (flat)</td>
              <td>{naive_m.get('rmse', 0):.4f}</td><td>baseline</td>
              <td>{naive_m.get('mae',  0):.4f}</td><td>baseline</td>
              <td>{naive_m.get('mape', 0):.2f}%</td><td>baseline</td>
              <td>{naive_m.get('dir_acc', 50):.1f}%</td><td>baseline</td>
            </tr>"""
        metrics_html = f"""
          <hr style="border:none;border-top:1px solid {BORDER};margin:16px 0 12px">
          <h3 style="margin-bottom:10px">Backtest Metrics
            <span class="note" style="font-size:11px;font-weight:normal;margin-left:8px">
              last {min(BACKTEST_DAYS, len(prices)-MIN_TRAIN_DAYS)} trading days,
              {horizon}-day steps
            </span>
          </h3>
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>RMSE</th><th></th>
                <th>MAE</th><th></th>
                <th>MAPE</th><th></th>
                <th>Dir. Acc.</th><th></th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <p class="note" style="margin-top:8px">
            Green = model beats the naive random-walk baseline on that metric.
            Directional accuracy = % of steps the model correctly predicted up vs down.
            A coin flip is 50 %. Beating naive RMSE is non-trivial under the EMH.
          </p>"""

    # ── Stat cards ─────────────────────────────────────────────────────────────
    def _stat_card(title, val, sub, clr=None):
        c = clr or TEXT_MAIN
        return (f'<div class="stat-card">'
                f'<div class="stat-val" style="color:{c}">{val}</div>'
                f'<div class="stat-sub">{title}</div>'
                f'<div class="stat-note">{sub}</div>'
                f'</div>')

    stat_cards = _stat_card(
        "Current Price",
        f"{currency} {last_price:,.2f}",
        f"as of {last_date}",
    )
    if arima_end is not None:
        stat_cards += _stat_card(
            f"ARIMA t+{horizon}",
            f"{currency} {arima_end:,.2f}",
            _fmt_pct(arima_chg),
            _color_delta(arima_chg),
        )
    if ets_end is not None:
        stat_cards += _stat_card(
            f"ETS t+{horizon}",
            f"{currency} {ets_end:,.2f}",
            _fmt_pct(ets_chg),
            _color_delta(ets_chg),
        )
    stat_cards += _stat_card(
        f"Naive t+{horizon}",
        f"{currency} {last_price:,.2f}",
        "flat (random walk)",
        TEXT_SUB,
    )

    # ── Plotly chart data ──────────────────────────────────────────────────────
    # Connect last historical point to first forecast point so the line is continuous
    conn_date  = hist_dates[-1:]
    conn_val   = [hist_vals[-1]]

    def _trace(name, x, y, color, dash="solid", width=2, fill=None,
               fill_color=None, show_legend=True, mode="lines"):
        t = {
            "type": "scatter", "mode": mode,
            "name": name, "x": x, "y": y,
            "line": {"color": color, "width": width, "dash": dash},
            "showlegend": show_legend,
        }
        if fill:
            t["fill"] = fill
            t["fillcolor"] = fill_color or (color + "22")
            t["line"]["width"] = 0
        return t

    fc_traces = []

    # Actual historical
    fc_traces.append({
        "type": "scatter", "mode": "lines",
        "name": "Actual", "x": hist_dates, "y": hist_vals,
        "line": {"color": TEXT_MAIN, "width": 1.5},
        "showlegend": True,
    })

    if do_arima and arima_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + arima_fc_px
        # 95 CI filled area
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ARIMA 95% CI", "x": fx, "y": [conn_val[0]] + arima_fc_hi95,
            "line": {"width": 0, "color": ACCENT_BLUE},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ARIMA 95% CI", "x": fx, "y": [conn_val[0]] + arima_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_BLUE + "22",
            "line": {"width": 0, "color": ACCENT_BLUE},
            "showlegend": False,
        })
        # 80 CI
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ARIMA 80% CI", "x": fx, "y": [conn_val[0]] + arima_fc_hi80,
            "line": {"width": 0, "color": ACCENT_BLUE},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ARIMA 80% CI", "x": fx, "y": [conn_val[0]] + arima_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_BLUE + "44",
            "line": {"width": 0, "color": ACCENT_BLUE},
            "showlegend": False,
        })
        # Forecast line
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": f"ARIMA{arima_order}", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_BLUE, "width": 2, "dash": "dot"},
            "showlegend": True,
        })

    if do_ets and ets_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + ets_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ETS 95% CI", "x": fx, "y": [conn_val[0]] + ets_fc_hi95,
            "line": {"width": 0, "color": ACCENT_AMB},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ETS 95% CI", "x": fx, "y": [conn_val[0]] + ets_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_AMB + "22",
            "line": {"width": 0, "color": ACCENT_AMB},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ETS 80% CI", "x": fx, "y": [conn_val[0]] + ets_fc_hi80,
            "line": {"width": 0, "color": ACCENT_AMB},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ETS 80% CI", "x": fx, "y": [conn_val[0]] + ets_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_AMB + "44",
            "line": {"width": 0, "color": ACCENT_AMB},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "ETS (damped)", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_AMB, "width": 2, "dash": "dot"},
            "showlegend": True,
        })

    if not do_arima or not arima_fc_px:
        # Show naive line
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Naive", "x": conn_date + future_strs,
            "y": conn_val + naive_fc_px,
            "line": {"color": TEXT_SUB, "width": 1, "dash": "dot"},
            "showlegend": True,
        })

    plotly_layout = {
        "paper_bgcolor": DARK_BG, "plot_bgcolor": DARK_BG,
        "font": {"color": TEXT_MAIN, "size": 11},
        "legend": {"bgcolor": CARD_BG, "bordercolor": BORDER,
                   "borderwidth": 1},
        "xaxis": {"gridcolor": BORDER, "zeroline": False,
                  "tickfont": {"color": TEXT_SUB}},
        "yaxis": {"gridcolor": BORDER, "zeroline": False,
                  "tickfont": {"color": TEXT_SUB},
                  "tickprefix": f"{currency} "},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 40},
        "hovermode": "x unified",
        "shapes": [{
            "type": "line",
            "x0": str(last_date), "x1": str(last_date),
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": TEXT_SUB, "width": 1, "dash": "dot"},
        }],
        "annotations": [{
            "x": str(last_date), "y": 1, "yref": "paper",
            "text": "← history | forecast →",
            "showarrow": False, "xanchor": "left",
            "font": {"color": TEXT_SUB, "size": 10},
        }],
    }

    # Backtest chart
    bt_traces_json = "[]"
    if bt and bt.get("dates"):
        bt_traces = [{
            "type": "scatter", "mode": "lines",
            "name": "Actual", "x": bt["dates"], "y": bt["actual"],
            "line": {"color": TEXT_MAIN, "width": 1.5},
        }]
        if do_arima and any(not np.isnan(v) for v in bt["arima_preds"]):
            bt_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"ARIMA{arima_order}",
                "x": bt["dates"], "y": bt["arima_preds"],
                "line": {"color": ACCENT_BLUE, "width": 1.5, "dash": "dot"},
            })
        if do_ets and any(not np.isnan(v) for v in bt["ets_preds"]):
            bt_traces.append({
                "type": "scatter", "mode": "lines",
                "name": "ETS",
                "x": bt["dates"], "y": bt["ets_preds"],
                "line": {"color": ACCENT_AMB, "width": 1.5, "dash": "dot"},
            })
        bt_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Naive",
            "x": bt["dates"], "y": bt["naive_preds"],
            "line": {"color": TEXT_SUB, "width": 1, "dash": "dot"},
        })
        bt_traces_json = json.dumps(bt_traces)

    # ACF chart
    acf_traces_json = "[]"
    if acf_vals:
        ci_line = 1.96 / np.sqrt(len(log_returns(prices)))
        acf_traces_json = json.dumps([
            {
                "type": "bar", "name": "ACF",
                "x": acf_lags, "y": acf_vals,
                "marker": {"color": [ACCENT_BLUE
                                     if abs(v) > ci_line else TEXT_SUB
                                     for v in acf_vals]},
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "95% CI",
                "x": [0, max(acf_lags) + 1],
                "y": [ci_line, ci_line],
                "line": {"color": ACCENT_RED, "dash": "dot", "width": 1},
                "showlegend": False,
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "−95% CI",
                "x": [0, max(acf_lags) + 1],
                "y": [-ci_line, -ci_line],
                "line": {"color": ACCENT_RED, "dash": "dot", "width": 1},
                "showlegend": False,
            },
        ])

    # Return distribution
    dist_traces_json = json.dumps([{
        "type": "bar", "name": "Log returns",
        "x": ret_hist_centers, "y": ret_hist_counts.tolist(),
        "marker": {"color": ACCENT_BLUE + "99"},
    }])

    # ── GARCH conditional vol chart ────────────────────────────────────────────
    garch_traces_json = "[]"
    garch_params_html = ""
    if garch_fit is not None and garch_var is not None:
        # Historical conditional vol (last hist_window days), annualised %
        hist_ann_vol = garch_hist_ann_vol(garch_fit)
        hist_ann_vol = hist_ann_vol[-hist_window:]
        g_hist_dates = hist_dates                        # same window as price chart

        # Forecast vol path: annualised % = sqrt(var) × 100 × sqrt(252)
        fcast_ann_vol = (np.sqrt(garch_var) * 100 * np.sqrt(252)).tolist()

        # Long-run annualised vol
        omega       = float(garch_fit.params.get("omega", 0))
        alpha       = float(garch_fit.params.get("alpha[1]",
                            garch_fit.params.get("alpha", 0)))
        beta_val    = float(garch_fit.params.get("beta[1]",
                            garch_fit.params.get("beta", 0)))
        persistence = alpha + beta_val
        lr_var_pct  = omega / max(1 - persistence, 1e-6)   # daily var in %² units
        lr_vol_ann  = float(np.sqrt(lr_var_pct) * np.sqrt(252))

        garch_traces_json = json.dumps([
            # History
            {
                "type": "scatter", "mode": "lines",
                "name": "Cond. vol (hist.)",
                "x": g_hist_dates,
                "y": [round(float(v), 3) for v in hist_ann_vol],
                "line": {"color": ACCENT_PRP, "width": 1.5},
                "showlegend": True,
            },
            # Forecast
            {
                "type": "scatter", "mode": "lines",
                "name": f"GARCH forecast ({horizon}d)",
                "x": future_strs,
                "y": [round(v, 3) for v in fcast_ann_vol],
                "line": {"color": ACCENT_PRP, "width": 2, "dash": "dot"},
                "showlegend": True,
            },
            # Long-run vol reference line (full x range)
            {
                "type": "scatter", "mode": "lines",
                "name": f"Long-run vol ({lr_vol_ann:.1f}%)",
                "x": [g_hist_dates[0], future_strs[-1]],
                "y": [lr_vol_ann, lr_vol_ann],
                "line": {"color": TEXT_SUB, "width": 1, "dash": "dot"},
                "showlegend": True,
            },
        ])

        # GARCH summary params for HTML card
        current_vol = float(hist_ann_vol[-1])
        aic_val = float(garch_fit.aic)
        bic_val = float(garch_fit.bic)
        ci_note = "GARCH" if use_garch_ci else "σ×√h"
        garch_params_html = f"""
        <div class="card" style="margin-top:16px">
          <h3>GARCH(1,1) Model Summary</h3>
          <table class="summary-table">
            <tr><td>Model</td><td>GARCH(1,1) · Normal innovations</td></tr>
            <tr><td>ω (base variance)</td><td>{omega:.6f}</td></tr>
            <tr><td>α (ARCH effect)</td><td>{alpha:.4f}</td></tr>
            <tr><td>β (GARCH effect)</td><td>{beta_val:.4f}</td></tr>
            <tr><td>Persistence (α+β)</td><td>{persistence:.4f}
              {"&nbsp;<span style='color:{ACCENT_RED}'>⚠ near unit-root</span>".format(ACCENT_RED=ACCENT_RED) if persistence > 0.98 else ""}</td></tr>
            <tr><td>Long-run ann. vol</td><td>{lr_vol_ann:.1f}%</td></tr>
            <tr><td>Current 1-day ann. vol</td><td>{current_vol:.1f}%</td></tr>
            <tr><td>AIC / BIC</td><td>{aic_val:.1f} / {bic_val:.1f}</td></tr>
            <tr><td>CI bands use</td><td>GARCH cumulative variance path</td></tr>
          </table>
          <p class="note" style="margin-top:10px">
            GARCH(1,1) models time-varying volatility: today's variance depends on
            yesterday's squared return (α) and yesterday's variance (β).
            High persistence means volatility shocks decay slowly.
            CI bands on the forecast chart use the h-step GARCH variance path
            rather than the flat σ×√h assumption, giving narrower bands in calm
            regimes and wider bands after large moves.
          </p>
        </div>"""

    compact_layout = {
        "paper_bgcolor": DARK_BG, "plot_bgcolor": DARK_BG,
        "font": {"color": TEXT_MAIN, "size": 10},
        "legend": {"bgcolor": CARD_BG, "bordercolor": BORDER, "borderwidth": 1},
        "xaxis": {"gridcolor": BORDER, "zeroline": False,
                  "tickfont": {"color": TEXT_SUB}},
        "yaxis": {"gridcolor": BORDER, "zeroline": False,
                  "tickfont": {"color": TEXT_SUB}},
        "margin": {"l": 50, "r": 10, "t": 10, "b": 40},
        "hovermode": "x unified",
        "showlegend": False,
    }

    # ── Render ─────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} Price Forecast — {today}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body  {{ background:{DARK_BG}; color:{TEXT_MAIN};
           font-family:'Inter',system-ui,sans-serif; font-size:13px;
           padding:20px; min-height:100vh; }}
  h1    {{ font-size:18px; font-weight:700; color:{TEXT_MAIN}; }}
  h2    {{ font-size:14px; font-weight:600; color:{TEXT_MAIN};
           text-transform:uppercase; letter-spacing:.06em; margin-bottom:12px; }}
  h3    {{ font-size:13px; font-weight:600; color:{TEXT_MAIN}; margin-bottom:10px; }}
  .header {{ display:flex; align-items:baseline; gap:12px; margin-bottom:20px; }}
  .sub    {{ color:{TEXT_SUB}; font-size:12px; }}
  .cards  {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat-card  {{ background:{CARD_BG}; border:1px solid {BORDER};
                 border-radius:8px; padding:14px 18px; min-width:140px; }}
  .stat-val   {{ font-size:20px; font-weight:700; line-height:1.2; }}
  .stat-sub   {{ font-size:11px; color:{TEXT_SUB}; margin-top:2px; }}
  .stat-note  {{ font-size:12px; font-weight:600; margin-top:4px; }}
  .card   {{ background:{CARD_BG}; border:1px solid {BORDER};
             border-radius:8px; padding:16px; margin-bottom:16px; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .chart-wrap {{ height:340px; }}
  .chart-wrap-sm {{ height:220px; }}
  .note  {{ font-size:11px; color:{TEXT_SUB}; font-style:italic; line-height:1.5; }}
  .disclaimer {{ background:#2a1a1a; border:1px solid #5a2020;
                 border-radius:8px; padding:14px; margin-top:16px;
                 font-size:11px; color:#e07070; line-height:1.6; }}
  .disclaimer b {{ color:#e05c5c; }}
  table  {{ width:100%; border-collapse:collapse; font-size:12px; }}
  td, th {{ padding:7px 10px; border-bottom:1px solid {BORDER};
             text-align:left; }}
  th     {{ color:{TEXT_SUB}; font-weight:600; font-size:11px;
             text-transform:uppercase; }}
  .metrics-table td:nth-child(1) {{ font-weight:500; }}
  .summary-table td:nth-child(1) {{ color:{TEXT_SUB}; width:40%; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
             font-size:10px; font-weight:600; letter-spacing:.04em;
             background:{ACCENT_BLUE}22; color:{ACCENT_BLUE}; }}
</style>
</head>
<body>

<div class="header">
  <h1>{ticker}</h1>
  <span class="sub">{info.get('name','')}</span>
  <span class="sub">·</span>
  <span class="sub">{info.get('sector','')}</span>
  <span style="margin-left:auto">
    <span class="badge">horizon {horizon}d</span>
    &nbsp;
    <span class="badge">{"ARIMA + ETS" if (do_arima and do_ets) else ("ARIMA" if do_arima else "ETS")}</span>
  </span>
</div>

<!-- Stat cards -->
<div class="cards">{stat_cards}</div>

<!-- Forecast chart -->
<div class="card">
  <h3>Price Forecast — {horizon}-Day Horizon</h3>
  <div class="chart-wrap" id="chart-forecast"></div>
  <p class="note" style="margin-top:8px">
    Shaded bands: inner = 80% CI, outer = 95% CI.
    {"CI bands use GARCH(1,1) cumulative conditional variance — wider after volatile periods, narrower in calm regimes." if use_garch_ci else "Uncertainty cone widens as σ × √h (random-walk approximation)."}
    Vertical line separates history from forecast.
  </p>
</div>

<!-- GARCH conditional vol chart -->
{"" if garch_fit is None else f'''
<div class="card">
  <h3>GARCH(1,1) Conditional Volatility <span class="note" style="font-size:11px;font-weight:normal;margin-left:8px">(annualised %)</span></h3>
  <div class="chart-wrap" id="chart-garch"></div>
  <p class="note" style="margin-top:8px">
    Historical conditional volatility estimated by GARCH(1,1) — captures volatility
    clustering (quiet and turbulent regimes). Dotted section = {horizon}-day ahead forecast.
    Dashed horizontal = long-run unconditional volatility.
    These are the variance estimates driving the CI bands in the forecast chart above.
  </p>
</div>
'''}

<!-- Backtest + metrics -->
{"" if not bt else f'''
<div class="card">
  <h3>Walk-Forward Backtest</h3>
  <div class="chart-wrap" id="chart-backtest"></div>
  <p class="note" style="margin-top:8px">
    Each segment = model trained on all data up to that point, predicting the next
    {horizon} days.  Gaps between segments reflect re-training windows.
  </p>
  {metrics_html}
</div>
'''}

<!-- Model summaries -->
<div class="two-col">
  {arima_summary_html}
  {ets_summary_html}
</div>
{garch_params_html}

<!-- Residual diagnostics -->
{"" if not acf_vals else f'''
<div class="two-col" style="margin-top:16px">
  <div class="card">
    <h3>ARIMA Residual ACF</h3>
    <div class="chart-wrap-sm" id="chart-acf"></div>
    <p class="note" style="margin-top:6px">
      Bars outside the red dashed lines (±1.96/√n) indicate residual autocorrelation —
      the model may be under-fitted.  Well-fitted ARIMA residuals should be white noise.
    </p>
  </div>
  <div class="card">
    <h3>Log-Return Distribution</h3>
    <div class="chart-wrap-sm" id="chart-dist"></div>
    <p class="note" style="margin-top:6px">
      Daily log-return histogram over the full training period.
      ARIMA assumes normally distributed innovations; heavy tails indicate fat-tail risk.
    </p>
  </div>
</div>
'''}

<!-- Disclaimer -->
<div class="disclaimer">
  <b>⚠ Important:</b> These forecasts are statistical model outputs, not investment advice.
  Financial markets are widely believed to follow a random walk under the
  Efficient Market Hypothesis — past prices contain limited information about future prices.
  ARIMA and ETS models may capture short-term autocorrelation or trend momentum but
  typically do <b>not</b> outperform naive baselines on a risk-adjusted basis over long horizons.
  Model outputs should be used for <b>research and educational purposes only</b>.
  Do not make investment decisions based solely on this output.
  Generated {today} · {ticker} · {info.get('name', '')}
</div>

<script>
const plotlyConfig = {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d','lasso2d'],
}};

// Forecast chart
(function() {{
  const traces = {json.dumps(fc_traces)};
  const layout = {json.dumps(plotly_layout)};
  Plotly.newPlot('chart-forecast', traces, layout, plotlyConfig);
}})();

// Backtest chart
(function() {{
  const traces = {bt_traces_json};
  if (traces.length && document.getElementById('chart-backtest')) {{
    const layout = {json.dumps({**plotly_layout,
                                "yaxis": {**plotly_layout["yaxis"],
                                          "tickprefix": f"{currency} "},
                                "annotations": [],
                                "shapes": []})};
    Plotly.newPlot('chart-backtest', traces, layout, plotlyConfig);
  }}
}})();

// ACF chart
(function() {{
  const traces = {acf_traces_json};
  if (traces.length && document.getElementById('chart-acf')) {{
    const layout = {json.dumps({**compact_layout,
                                "yaxis": {**compact_layout["yaxis"],
                                          "range": [-1, 1]}})};
    Plotly.newPlot('chart-acf', traces, layout, plotlyConfig);
  }}
}})();

// Distribution chart
(function() {{
  const traces = {dist_traces_json};
  if (document.getElementById('chart-dist')) {{
    Plotly.newPlot('chart-dist', traces, {json.dumps(compact_layout)}, plotlyConfig);
  }}
}})();

// GARCH conditional vol chart
(function() {{
  const traces = {garch_traces_json};
  if (traces.length && document.getElementById('chart-garch')) {{
    const layout = {json.dumps({
        **compact_layout,
        "showlegend": True,
        "legend": {"bgcolor": CARD_BG, "bordercolor": BORDER, "borderwidth": 1},
        "yaxis": {**compact_layout["yaxis"], "ticksuffix": "%"},
        "xaxis": {**compact_layout["xaxis"]},
        "hovermode": "x unified",
        "shapes": [{
            "type": "line",
            "x0": str(last_date), "x1": str(last_date),
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": TEXT_SUB, "width": 1, "dash": "dot"},
        }],
    })};
    Plotly.newPlot('chart-garch', traces, layout, plotlyConfig);
  }}
}})();
</script>
</body>
</html>
"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Price / return forecasting tool")
    ap.add_argument("--ticker",  default="AAPL")
    ap.add_argument("--horizon", type=int, default=30,
                    help="Trading days to forecast (default 30)")
    ap.add_argument("--model",   default="arima",
                    choices=["arima", "ets", "both"])
    ap.add_argument("--period",  default="5y",
                    choices=["2y", "5y", "10y"])
    args = ap.parse_args()

    ticker  = args.ticker.upper().strip()
    horizon = max(5, min(args.horizon, 252))
    do_arima = args.model in ("arima", "both")
    do_ets   = args.model in ("ets",   "both")

    if do_arima and not PMDARIMA_OK:
        print("[warn] pmdarima not installed — falling back to ETS only.")
        do_arima = False
        do_ets   = True

    print(f"\n{'='*60}", flush=True)
    print(f"  Price Forecast · {ticker} · horizon={horizon}d · "
          f"model={'ARIMA+ETS' if (do_arima and do_ets) else ('ARIMA' if do_arima else 'ETS')}",
          flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Data
    prices = fetch_prices(ticker, args.period)
    info   = fetch_info(ticker)

    n_steps = (2 if do_arima else 0) + (1 if do_ets else 0) + 2  # GARCH + backtest
    step = [0]
    def _step(label):
        step[0] += 1
        print(f"\n[{step[0]}/{n_steps}] {label}", flush=True)

    ret_all = log_returns(prices)

    # 2. ARIMA
    arima_fit        = None
    arima_order      = None
    arima_resid_std  = 0.0
    if do_arima:
        _step("ARIMA")
        arima_order, arima_resid_std, arima_fit = fit_arima(ret_all)

    # 3. ETS
    ets_fit       = None
    ets_resid_std = 0.0
    if do_ets:
        _step("ETS")
        log_px = np.log(prices)
        ets_fit = fit_ets(log_px)
        ets_resid_std = float(np.std(ets_fit.resid))

    # 4. GARCH (always, when arch is installed) — volatility model
    garch_fit_ = None
    garch_var_ = None
    if ARCH_OK:
        _step("GARCH(1,1)")
        garch_fit_ = fit_garch(ret_all)
        garch_var_ = garch_var_path(garch_fit_, horizon)

    # 5. Backtest
    _step("Backtest")
    bt = backtest(prices, horizon, arima_order, do_arima, do_ets)
    if not bt:
        print("  [warn] Not enough data for backtest (need ≥504 trading days).",
              flush=True)

    # 6. HTML
    print("\n  Building HTML report…", flush=True)
    html = build_html(
        ticker=ticker,
        info=info,
        prices=prices,
        horizon=horizon,
        model_arg=args.model,
        arima_fit=arima_fit,
        arima_order=arima_order,
        arima_resid_std=arima_resid_std,
        ets_fit=ets_fit,
        ets_resid_std=ets_resid_std,
        bt=bt,
        do_arima=do_arima,
        do_ets=do_ets,
        garch_fit=garch_fit_,
        garch_var=garch_var_,
    )

    today    = datetime.date.today().strftime("%Y_%m_%d")
    out_path = os.path.join(OUTPUT_DIR, f"{today}_{ticker}_forecast.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  ✓ Saved → {out_path}", flush=True)
    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open(f"file://{out_path}")
        print("  ✓ Opened in browser\n", flush=True)


if __name__ == "__main__":
    main()
