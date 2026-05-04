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

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

import argparse
import datetime
import json
import os
import sys
import warnings
import webbrowser

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")          # suppress all Python warnings globally
warnings.filterwarnings("ignore")        # belt-and-suspenders for any re-registrations

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

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[warn] xgboost not found — XGBoost disabled.  pip install xgboost")

try:
    from prophet import Prophet as _Prophet
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False
    print("[warn] prophet not found — Prophet disabled.  pip install prophet")

try:
    import torch as _torch
    import torch.nn as _nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[warn] torch not found — LSTM disabled.  pip install torch")

try:
    import timesfm as _timesfm
    TIMESFM_OK = True
except ImportError:
    TIMESFM_OK = False

try:
    from chronos import BaseChronosPipeline as _ChronosPipeline
    CHRONOS_OK = True
except ImportError:
    CHRONOS_OK = False

try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    import lightning.pytorch as _pl
    TFT_OK = True
except ImportError:
    TFT_OK = False

# N-HiTS and N-BEATS are implemented natively in PyTorch below (no external library
# needed) — they run whenever TORCH_OK is True.
NEURALFORECAST_OK = TORCH_OK  # alias used by build_html / main

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as _acf

import logging as _logging
_logging.getLogger("statsmodels").setLevel(_logging.ERROR)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "forecastData")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allow importing from the project root (valuation_models.py lives there)
sys.path.insert(0, os.path.dirname(ROOT))

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
ACCENT_TEA  = "#2dd4bf"   # teal — composite fundamental target
ACCENT_ORG  = "#fb923c"   # Prophet
ACCENT_FCS  = "#e879f9"   # LSTM
ACCENT_CYN  = "#22d3ee"   # TimesFM
ACCENT_LIM  = "#a3e635"   # Chronos
ACCENT_GBM  = "#f43f5e"   # rose — Monte Carlo GBM
ACCENT_TFT  = "#818cf8"   # indigo — TFT
ACCENT_NHT  = "#f97316"   # orange — N-HiTS
ACCENT_NBT  = "#06b6d4"   # sky — N-BEATS

# Feature column names produced by build_features() — shared by XGB and LSTM
FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rsi14", "sma_ratio", "rvol_z", "rvol_20d",
]

MACRO_FEATURE_COLS = [
    "vix_level", "vix_5d_change", "yield_10y",
    "sector_rel_21d", "market_rel_21d",
]

_SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}

_MACRO_CACHE: dict = {}   # keyed by (sector, period)

# Sector-level median multiples used when live peer data is unavailable.
# These are approximate market-cycle medians; update periodically.
_SECTOR_BM = {
    "Technology":             {"pe": 28, "pfcf": 30, "ev_ebitda": 22},
    "Communication Services": {"pe": 22, "pfcf": 22, "ev_ebitda": 16},
    "Consumer Discretionary": {"pe": 22, "pfcf": 20, "ev_ebitda": 14},
    "Consumer Staples":       {"pe": 20, "pfcf": 20, "ev_ebitda": 13},
    "Health Care":            {"pe": 22, "pfcf": 22, "ev_ebitda": 16},
    "Financials":             {"pe": 13, "pfcf": 14, "ev_ebitda": 10},
    "Industrials":            {"pe": 20, "pfcf": 18, "ev_ebitda": 14},
    "Energy":                 {"pe": 12, "pfcf": 11, "ev_ebitda":  8},
    "Materials":              {"pe": 16, "pfcf": 14, "ev_ebitda": 10},
    "Real Estate":            {"pe": 30, "pfcf": 28, "ev_ebitda": 20},
    "Utilities":              {"pe": 17, "pfcf": 15, "ev_ebitda": 12},
}
_DEFAULT_BM = {"pe": 20, "pfcf": 20, "ev_ebitda": 14}


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost Forecast
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def build_features(prices: pd.Series, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build feature matrix from price series.
    Features: log returns at 1/3/5/10/20/60d lags, RSI(14),
              SMA20/SMA50 ratio, vol z-score (20d), realized vol (20d).
    """
    lr = np.log(prices / prices.shift(1))
    features = {}
    for lag in (1, 3, 5, 10, 20, 60):
        features[f"ret_{lag}d"] = lr.shift(1).rolling(lag).sum()
    features["rsi14"]     = _rsi(prices)
    features["sma_ratio"] = prices.rolling(20).mean() / prices.rolling(50).mean()
    _rvol = lr.rolling(20).std() * (252 ** 0.5)
    _rvol_mean = _rvol.rolling(252).mean()
    _rvol_std  = _rvol.rolling(252).std().replace(0, float("nan"))
    features["rvol_z"]    = (_rvol - _rvol_mean) / _rvol_std
    features["rvol_20d"]  = _rvol
    df = pd.DataFrame(features, index=prices.index)
    if macro_df is not None:
        for col in MACRO_FEATURE_COLS:
            if col in macro_df.columns:
                df[col] = macro_df[col].reindex(df.index)
    return df


def fit_xgb_forecast(prices: pd.Series, horizon: int,
                     backtest_window: int = 252,
                     macro_df: pd.DataFrame = None) -> dict:
    """
    Walk-forward XGBoost forecast of horizon-day cumulative log return.
    Uses expanding window, refits every 60 days.
    Returns: forecast_prices (len=horizon), ci_80_lo, ci_80_hi,
             ci_95_lo, ci_95_hi, backtest_mape (float or None).
    """
    if not XGB_OK:
        return {}

    feats = build_features(prices, macro_df=macro_df)
    lr    = np.log(prices / prices.shift(1))

    # Target: horizon-day forward cumulative log return
    target = lr.rolling(horizon).sum().shift(-horizon)

    df = feats.copy()
    df["target"] = target
    df = df.dropna()

    if len(df) < backtest_window + horizon + 60:
        # Not enough data — just fit on all available history
        X_train = df.drop(columns=["target"]).values
        y_train = df["target"].values
        model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                             max_depth=4, subsample=0.8, random_state=42,
                             verbosity=0)
        model.fit(X_train, y_train)
        # Point forecast from last row
        last_feats = feats.iloc[[-1]].dropna(axis=1)
        X_last = last_feats.reindex(columns=feats.dropna().columns, fill_value=0).values
        pred_lr = float(model.predict(X_last)[0])
        # CI: use residual std as proxy for uncertainty
        resid_std = float(np.std(y_train - model.predict(X_train)))
        z80, z95 = 1.282, 1.960
        last_price = float(prices.iloc[-1])
        future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
        # Linear interpolation of price path from last_price to forecast endpoint
        pred_end  = last_price * np.exp(pred_lr)
        lo80_end  = last_price * np.exp(pred_lr - z80 * resid_std)
        hi80_end  = last_price * np.exp(pred_lr + z80 * resid_std)
        lo95_end  = last_price * np.exp(pred_lr - z95 * resid_std)
        hi95_end  = last_price * np.exp(pred_lr + z95 * resid_std)
        t = np.linspace(0, 1, horizon)
        forecast_prices = last_price + t * (pred_end - last_price)
        ci_80_lo        = last_price + t * (lo80_end - last_price)
        ci_80_hi        = last_price + t * (hi80_end - last_price)
        ci_95_lo        = last_price + t * (lo95_end - last_price)
        ci_95_hi        = last_price + t * (hi95_end - last_price)
        return {
            "forecast": forecast_prices.tolist(),
            "ci_80_lo": ci_80_lo.tolist(), "ci_80_hi": ci_80_hi.tolist(),
            "ci_95_lo": ci_95_lo.tolist(), "ci_95_hi": ci_95_hi.tolist(),
            "backtest_mape": None,
            "future_dates": [str(d.date()) for d in future_dates],
        }

    # Walk-forward backtest + final forecast
    bt_start  = len(df) - backtest_window
    actuals, preds = [], []
    model = None
    for i in range(bt_start, len(df)):
        if (i - bt_start) % 60 == 0:
            # Refit on all data up to current point
            X_tr = df.iloc[:i].drop(columns=["target"]).values
            y_tr = df.iloc[:i]["target"].values
            model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=4, subsample=0.8, random_state=42,
                                 verbosity=0)
            model.fit(X_tr, y_tr)
        X_i = df.iloc[[i]].drop(columns=["target"]).values
        preds.append(float(model.predict(X_i)[0]))
        actuals.append(float(df.iloc[i]["target"]))

    # Backtest MAPE on log-returns
    valid = [(a, p) for a, p in zip(actuals, preds) if abs(a) > 1e-6]
    bt_mape = float(np.mean([abs(a - p) / abs(a) * 100 for a, p in valid])) if valid else None

    # Residual std for CI
    resid_std = float(np.std([a - p for a, p in zip(actuals, preds)]))

    # Final forecast using all data
    X_all  = df.drop(columns=["target"]).values
    y_all  = df["target"].values
    model_final = XGBRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=4, subsample=0.8, random_state=42,
                               verbosity=0)
    model_final.fit(X_all, y_all)
    last_row = feats.iloc[[-1]].reindex(columns=df.drop(columns=["target"]).columns, fill_value=0)
    pred_lr  = float(model_final.predict(last_row.values)[0])

    z80, z95  = 1.282, 1.960
    last_price = float(prices.iloc[-1])
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
    pred_end = last_price * np.exp(pred_lr)
    lo80_end = last_price * np.exp(pred_lr - z80 * resid_std)
    hi80_end = last_price * np.exp(pred_lr + z80 * resid_std)
    lo95_end = last_price * np.exp(pred_lr - z95 * resid_std)
    hi95_end = last_price * np.exp(pred_lr + z95 * resid_std)
    t = np.linspace(0, 1, horizon)
    forecast_prices = last_price + t * (pred_end - last_price)
    ci_80_lo        = last_price + t * (lo80_end - last_price)
    ci_80_hi        = last_price + t * (hi80_end - last_price)
    ci_95_lo        = last_price + t * (lo95_end - last_price)
    ci_95_hi        = last_price + t * (hi95_end - last_price)
    return {
        "forecast": forecast_prices.tolist(),
        "ci_80_lo": ci_80_lo.tolist(), "ci_80_hi": ci_80_hi.tolist(),
        "ci_95_lo": ci_95_lo.tolist(), "ci_95_hi": ci_95_hi.tolist(),
        "backtest_mape": bt_mape,
        "future_dates": [str(d.date()) for d in future_dates],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prophet Forecast
# ══════════════════════════════════════════════════════════════════════════════

def fit_prophet_forecast(prices: pd.Series, horizon: int,
                         backtest_window: int = 252) -> dict:
    """
    Facebook Prophet additive model on log prices.
    Handles weekly / yearly seasonality automatically.
    Walk-forward backtest (horizon-day steps) for MAPE; final model uses all data.
    Returns standard forecast dict.
    """
    if not PROPHET_OK:
        return {}

    import os as _os
    import io as _io

    log_px = np.log(prices)
    df_all = pd.DataFrame({"ds": log_px.index, "y": log_px.values})

    last_price  = float(prices.iloc[-1])
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
    future_strs  = [str(d.date()) for d in future_dates]

    def _make_prophet(uncertainty_samples=0):
        return _Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            uncertainty_samples=uncertainty_samples,
            changepoint_prior_scale=0.05,
        )

    def _fit_quiet(m, df):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m.fit(df)
        return m

    # Walk-forward backtest for MAPE
    bt_mape = None
    n = len(df_all)
    if n >= MIN_TRAIN_DAYS + backtest_window + horizon:
        bt_start = n - backtest_window
        n_steps  = max(1, backtest_window // horizon)
        actuals, preds = [], []
        for step in range(n_steps):
            t0 = bt_start + step * horizon
            t1 = t0 + horizon
            if t1 > n:
                break
            try:
                m = _make_prophet(0)
                _fit_quiet(m, df_all.iloc[:t0])
                fut = m.make_future_dataframe(periods=horizon, freq="B")
                fc  = m.predict(fut)
                pred_log = fc.iloc[-horizon:]["yhat"].values
                actual_log = df_all.iloc[t0:t1]["y"].values
                actuals.extend(actual_log.tolist())
                preds.extend(pred_log.tolist())
            except Exception:
                pass
        if actuals:
            valid = [(a, p) for a, p in zip(actuals, preds) if abs(a) > 1e-6]
            bt_mape = float(np.mean([abs(a - p) / abs(a) * 100 for a, p in valid])) if valid else None

    # Final forecast with uncertainty
    try:
        m_final = _make_prophet(uncertainty_samples=500)
        _fit_quiet(m_final, df_all)
        fut = m_final.make_future_dataframe(periods=horizon, freq="B")
        fc  = m_final.predict(fut)
        tail = fc.iloc[-horizon:]

        fc_log     = tail["yhat"].values
        lo80_log   = tail["yhat_lower"].values  # prophet default interval_width=0.8
        hi80_log   = tail["yhat_upper"].values

        # Approximate 95% CI by scaling the 80% bands
        half80 = (hi80_log - lo80_log) / 2
        scale  = CI_95_Z / CI_80_Z
        lo95_log = fc_log - half80 * scale
        hi95_log = fc_log + half80 * scale

        # Back-transform from log-price to price space
        fc_px    = np.exp(fc_log).tolist()
        lo80_px  = np.exp(lo80_log).tolist()
        hi80_px  = np.exp(hi80_log).tolist()
        lo95_px  = np.exp(lo95_log).tolist()
        hi95_px  = np.exp(hi95_log).tolist()

        return {
            "forecast": fc_px, "ci_80_lo": lo80_px, "ci_80_hi": hi80_px,
            "ci_95_lo": lo95_px, "ci_95_hi": hi95_px,
            "backtest_mape": bt_mape,
            "future_dates": future_strs,
        }
    except Exception as exc:
        print(f"  [warn] Prophet final forecast failed: {exc}", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Forecast
# ══════════════════════════════════════════════════════════════════════════════

class _LSTMForecaster(_nn.Module if TORCH_OK else object):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.25):
        if not TORCH_OK:
            return
        super().__init__()
        self.lstm = _nn.LSTM(
            n_features, hidden, n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = _nn.Dropout(dropout)
        self.head    = _nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def fit_lstm_forecast(prices: pd.Series, horizon: int,
                      backtest_window: int = 252,
                      macro_df: pd.DataFrame = None) -> dict:
    """
    2-layer LSTM with MC-Dropout CI.
    Features: 9-dim from build_features() reused.
    Sequence length: 20 trading days.
    Target: horizon-day cumulative log-return.
    """
    if not TORCH_OK:
        return {}

    SEQ_LEN   = 20
    HIDDEN    = 32      # kept small for CPU speed
    N_LAYERS  = 2
    EPOCHS    = 60      # early stopping usually kicks in well before this
    BATCH     = 64
    LR        = 5e-4
    WD        = 1e-5
    MIN_ROWS = SEQ_LEN + horizon + 60   # bare minimum

    # Build feature matrix (same pipeline as XGB)
    feat_df = build_features(prices, macro_df=macro_df)
    lr      = np.log(prices / prices.shift(1))
    target  = lr.rolling(horizon).sum().shift(-horizon)

    df = feat_df.copy()
    df["target"] = target
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    if len(df) < MIN_ROWS:
        return {}

    X_vals = df[FEATURE_COLS].values.astype(np.float32)
    y_vals = df["target"].values.astype(np.float32)

    # Standardise features on the full training set
    mu  = X_vals.mean(axis=0)
    sig = X_vals.std(axis=0) + 1e-8
    X_norm = (X_vals - mu) / sig

    # Build sequences: each sample is SEQ_LEN consecutive rows predicting the last row's target
    def _make_seqs(X, y):
        seqs, tgts = [], []
        for i in range(SEQ_LEN, len(X)):
            seqs.append(X[i - SEQ_LEN:i])
            tgts.append(y[i - 1])   # target aligned to last day in window
        return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)

    X_seq, y_seq = _make_seqs(X_norm, y_vals)
    if len(X_seq) < 100:
        return {}

    # Train/val split (80/20 by time)
    split      = int(len(X_seq) * 0.8)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_seq[:split], y_seq[split:]

    X_tr_t  = _torch.tensor(X_tr)
    y_tr_t  = _torch.tensor(y_tr)
    X_val_t = _torch.tensor(X_val)
    y_val_t = _torch.tensor(y_val)

    model     = _LSTMForecaster(len(FEATURE_COLS), HIDDEN, N_LAYERS)
    optimizer = _torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = _nn.MSELoss()

    best_val, best_state, patience, counter = float("inf"), None, 15, 0
    for epoch in range(EPOCHS):
        model.train()
        idx = _torch.randperm(len(X_tr_t))
        for b in range(0, len(X_tr_t), BATCH):
            bi = idx[b:b + BATCH]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[bi]), y_tr_t[bi])
            loss.backward()
            _torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with _torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Residual std on validation set for CI
    model.eval()
    with _torch.no_grad():
        val_pred = model(X_val_t).numpy()
    resid_std = float(np.std(y_val - val_pred))

    # Backtest MAPE on validation
    valid = [(float(a), float(p)) for a, p in zip(y_val, val_pred) if abs(a) > 1e-6]
    bt_mape = float(np.mean([abs(a - p) / abs(a) * 100 for a, p in valid])) if valid else None

    # Retrain on all data (with early stopping against training loss to avoid infinite run)
    model2   = _LSTMForecaster(len(FEATURE_COLS), HIDDEN, N_LAYERS)
    opt2     = _torch.optim.Adam(model2.parameters(), lr=LR, weight_decay=WD)
    X_all_t  = _torch.tensor(X_seq)
    y_all_t  = _torch.tensor(y_seq)
    best_tr, no_improve = float("inf"), 0
    for _ in range(EPOCHS):
        model2.train()
        idx2 = _torch.randperm(len(X_all_t))
        for b in range(0, len(X_all_t), BATCH):
            bi = idx2[b:b + BATCH]
            opt2.zero_grad()
            loss = criterion(model2(X_all_t[bi]), y_all_t[bi])
            loss.backward()
            _torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            opt2.step()
        with _torch.no_grad():
            tr_loss = criterion(model2(X_all_t), y_all_t).item()
        if tr_loss < best_tr - 1e-6:
            best_tr, no_improve = tr_loss, 0
        else:
            no_improve += 1
            if no_improve >= 15:
                break

    # Current snapshot: last SEQ_LEN rows of features
    last_feats = X_norm[-SEQ_LEN:]
    last_t     = _torch.tensor(last_feats[np.newaxis])   # (1, SEQ_LEN, n_feat)

    # MC Dropout CI: 200 stochastic passes
    model2.train()   # keep dropout active
    mc_preds = []
    with _torch.no_grad():
        for _ in range(200):
            mc_preds.append(float(model2(last_t).item()))
    pred_lr  = float(np.mean(mc_preds))
    mc_std   = float(np.std(mc_preds))
    ci_std   = max(mc_std, resid_std)   # take wider of the two

    last_price   = float(prices.iloc[-1])
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
    future_strs  = [str(d.date()) for d in future_dates]

    pred_end = last_price * np.exp(pred_lr)
    lo80_end = last_price * np.exp(pred_lr - CI_80_Z * ci_std)
    hi80_end = last_price * np.exp(pred_lr + CI_80_Z * ci_std)
    lo95_end = last_price * np.exp(pred_lr - CI_95_Z * ci_std)
    hi95_end = last_price * np.exp(pred_lr + CI_95_Z * ci_std)

    t = np.linspace(0, 1, horizon)
    return {
        "forecast":       (last_price + t * (pred_end - last_price)).tolist(),
        "ci_80_lo":       (last_price + t * (lo80_end - last_price)).tolist(),
        "ci_80_hi":       (last_price + t * (hi80_end - last_price)).tolist(),
        "ci_95_lo":       (last_price + t * (lo95_end - last_price)).tolist(),
        "ci_95_hi":       (last_price + t * (hi95_end - last_price)).tolist(),
        "backtest_mape":  bt_mape,
        "future_dates":   future_strs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TimesFM Forecast (Google, zero-shot)
# ══════════════════════════════════════════════════════════════════════════════

_TIMESFM_MODEL = None   # lazy-loaded singleton

def fit_timesfm_forecast(prices: pd.Series, horizon: int) -> dict:
    """
    Zero-shot forecast using Google TimesFM-1.0-200M (PyTorch backend).
    Downloads ~200 MB on first use; cached in HuggingFace cache thereafter.
    Returns standard forecast dict with quantile-derived CI bands.
    """
    if not TIMESFM_OK:
        return {}

    global _TIMESFM_MODEL

    last_price   = float(prices.iloc[-1])
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
    future_strs  = [str(d.date()) for d in future_dates]

    # Use log prices as input — TimesFM works on the level series
    log_px = np.log(prices.values).tolist()
    # Limit context to 512 points (model's effective context window)
    context = log_px[-512:]

    try:
        if _TIMESFM_MODEL is None:
            print("  Loading TimesFM-1.0-200M (downloading on first run)…", flush=True)
            _TIMESFM_MODEL = _timesfm.TimesFm(
                hparams=_timesfm.TimesFmHparams(
                    backend="torch",
                    per_core_batch_size=32,
                    horizon_len=horizon,
                    num_layers=20,
                    model_dims=1280,
                ),
                checkpoint=_timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )
            _TIMESFM_MODEL.initialize_from_checkpoint()

        point_fc, quantile_fc = _TIMESFM_MODEL.forecast(
            [context],
            freq=[0],   # 0 = high-frequency (daily)
        )
        # point_fc shape: (1, horizon)  quantile_fc shape: (1, horizon, n_quantiles)
        # default quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fc_log   = point_fc[0]
        q_fc     = quantile_fc[0]   # (horizon, n_quantiles)

        # Back-transform to price space
        fc_px    = np.exp(fc_log).tolist()
        lo80_px  = np.exp(q_fc[:, 1]).tolist()  # 20th pct
        hi80_px  = np.exp(q_fc[:, 7]).tolist()  # 80th pct
        lo95_px  = np.exp(q_fc[:, 0]).tolist()  # 10th pct
        hi95_px  = np.exp(q_fc[:, 8]).tolist()  # 90th pct

        return {
            "forecast": fc_px, "ci_80_lo": lo80_px, "ci_80_hi": hi80_px,
            "ci_95_lo": lo95_px, "ci_95_hi": hi95_px,
            "backtest_mape": None,   # zero-shot — no backtest
            "future_dates": future_strs,
        }
    except Exception as exc:
        print(f"  [warn] TimesFM forecast failed: {exc}", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Chronos Forecast (Amazon, zero-shot)
# ══════════════════════════════════════════════════════════════════════════════

_CHRONOS_PIPELINE = None   # lazy-loaded singleton

def fit_chronos_forecast(prices: pd.Series, horizon: int) -> dict:
    """
    Zero-shot probabilistic forecast using Amazon Chronos-Bolt-Small.
    Downloads ~300 MB on first use; cached in HuggingFace cache thereafter.
    Returns standard forecast dict with sample-derived CI bands.
    """
    if not CHRONOS_OK:
        return {}

    global _CHRONOS_PIPELINE

    last_price   = float(prices.iloc[-1])
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
    future_strs  = [str(d.date()) for d in future_dates]

    # Use log prices so the model operates on smoother values
    log_px = np.log(prices.values[-512:]).astype(np.float32)
    context = _torch.tensor(log_px).unsqueeze(0)   # (1, context_len)

    try:
        if _CHRONOS_PIPELINE is None:
            print("  Loading Chronos-Bolt-Small (downloading on first run)…", flush=True)
            _CHRONOS_PIPELINE = _ChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                device_map="cpu",
                torch_dtype=_torch.float32,
            )

        # Returns (batch=1, num_samples, horizon)
        samples = _CHRONOS_PIPELINE.predict(
            context=context,
            prediction_length=horizon,
            num_samples=200,
        )
        # samples shape: (1, 200, horizon) — in log-price space
        s = samples[0].numpy()   # (200, horizon)

        # Compute median + quantiles
        fc_log   = np.median(s, axis=0)
        lo80_log = np.percentile(s, 10, axis=0)
        hi80_log = np.percentile(s, 90, axis=0)
        lo95_log = np.percentile(s,  5, axis=0)
        hi95_log = np.percentile(s, 95, axis=0)

        # Back-transform to price space
        fc_px   = np.exp(fc_log).tolist()
        lo80_px = np.exp(lo80_log).tolist()
        hi80_px = np.exp(hi80_log).tolist()
        lo95_px = np.exp(lo95_log).tolist()
        hi95_px = np.exp(hi95_log).tolist()

        return {
            "forecast": fc_px, "ci_80_lo": lo80_px, "ci_80_hi": hi80_px,
            "ci_95_lo": lo95_px, "ci_95_hi": hi95_px,
            "backtest_mape": None,
            "future_dates": future_strs,
        }
    except Exception as exc:
        print(f"  [warn] Chronos forecast failed: {exc}", flush=True)
        return {}


def fit_montecarlo_forecast(prices: pd.Series, horizon: int,
                             n_sims: int = 1000) -> dict:
    """
    Geometric Brownian Motion Monte Carlo simulation.
    Estimates drift and volatility from log-returns, simulates n_sims paths.
    CI derived from cross-path percentiles at each future step.
    """
    try:
        lr  = np.log(prices / prices.shift(1)).dropna().values
        mu  = float(np.mean(lr))
        sig = float(np.std(lr, ddof=1))
        last_price = float(prices.iloc[-1])
        # Vectorised simulation
        Z    = np.random.randn(n_sims, horizon)
        step = (mu - 0.5 * sig ** 2) + sig * Z
        paths = last_price * np.exp(np.cumsum(step, axis=1))
        forecast = np.median(paths, axis=0).tolist()
        ci_80_lo = np.percentile(paths, 10, axis=0).tolist()
        ci_80_hi = np.percentile(paths, 90, axis=0).tolist()
        ci_95_lo = np.percentile(paths,  5, axis=0).tolist()
        ci_95_hi = np.percentile(paths, 95, axis=0).tolist()
        # Walk-forward MAPE (lightweight: 200 sims per window step)
        bt_window = min(252, len(lr) - horizon - 60)
        mapes = []
        if bt_window > 0:
            start = len(lr) - bt_window - horizon
            for i in range(0, bt_window, horizon):
                idx = start + i
                if idx + horizon >= len(prices):
                    break
                sub_lr = lr[:idx]
                if len(sub_lr) < 20:
                    continue
                m_bt = float(np.mean(sub_lr))
                s_bt = float(np.std(sub_lr, ddof=1))
                sp   = float(prices.iloc[idx])
                Z_bt = np.random.randn(200, horizon)
                st_bt = (m_bt - 0.5 * s_bt**2) + s_bt * Z_bt
                p_bt  = sp * np.exp(np.cumsum(st_bt, axis=1))
                pred_end = float(np.median(p_bt[:, -1]))
                actual_end = float(prices.iloc[idx + horizon])
                if actual_end > 0:
                    mapes.append(abs(pred_end - actual_end) / actual_end * 100)
        bt_mape = float(np.mean(mapes)) if mapes else None
        future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
        print(f"  Monte Carlo MAPE: {bt_mape:.2f}%" if bt_mape else "  Monte Carlo: no backtest MAPE", flush=True)
        return {
            "forecast":      forecast,
            "ci_80_lo":      ci_80_lo, "ci_80_hi": ci_80_hi,
            "ci_95_lo":      ci_95_lo, "ci_95_hi": ci_95_hi,
            "backtest_mape": bt_mape,
            "future_dates":  [str(d.date()) for d in future_dates],
        }
    except Exception as exc:
        print(f"  [warn] Monte Carlo failed: {exc}", flush=True)
        return {}


def fit_tft_forecast(prices: pd.Series, horizon: int,
                     macro_df: pd.DataFrame = None) -> dict:
    """
    Temporal Fusion Transformer (pytorch-forecasting).
    Small architecture (hidden=16, 1 LSTM layer, 30 epochs) on last 500 days.
    Quantile outputs give native CI. No walk-forward backtest (training cost).
    """
    if not TFT_OK:
        return {}
    try:
        log_px = np.log(prices.values.astype(float))
        n = min(len(log_px), 500)
        log_px = log_px[-n:]
        px_sub = prices.iloc[-n:]
        df_tft = pd.DataFrame({
            "log_price": log_px,
            "time_idx":  np.arange(n),
            "group":     "A",
        }, index=px_sub.index)
        # Add macro covariates if available
        time_varying_unknown = ["log_price"]
        if macro_df is not None:
            mac = macro_df[MACRO_FEATURE_COLS].reindex(px_sub.index, method="ffill").fillna(0)
            for col in MACRO_FEATURE_COLS:
                df_tft[col] = mac[col].values
            time_varying_unknown += MACRO_FEATURE_COLS
        max_enc = 60
        max_pred = horizon
        train_cutoff = n - max_pred - 1
        if train_cutoff < max_enc + 10:
            return {}
        from pytorch_forecasting.data import EncoderNormalizer as _EncNorm
        training = TimeSeriesDataSet(
            df_tft[df_tft.time_idx <= train_cutoff],
            time_idx="time_idx",
            target="log_price",
            group_ids=["group"],
            max_encoder_length=max_enc,
            max_prediction_length=max_pred,
            time_varying_unknown_reals=time_varying_unknown,
            time_varying_known_reals=["time_idx"],
            # EncoderNormalizer centers each sequence on its encoder mean/std so
            # the model sees stationary residuals; predict() back-transforms automatically
            target_normalizer=_EncNorm(transformation="softplus"),
        )
        val_ds   = TimeSeriesDataSet.from_dataset(training, df_tft, predict=True, stop_randomization=True)
        train_dl = training.to_dataloader(train=True,  batch_size=64, num_workers=0)
        val_dl   = val_ds.to_dataloader(  train=False, batch_size=64, num_workers=0)
        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=16, lstm_layers=1, dropout=0.1,
            output_size=7,
            loss=QuantileLoss(quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
        )
        print("  Training TFT (30 epochs)…", flush=True)
        import io as _io, sys as _sys
        _old_stderr = _sys.stderr
        _sys.stderr = _io.StringIO()   # silence lightning/tensorboard subprocess noise
        try:
            trainer = _pl.Trainer(
                max_epochs=30,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[],
            )
            trainer.fit(tft_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            # predict() returns log-price values back-transformed from EncoderNormalizer
            preds = tft_model.predict(val_dl, mode="quantiles", return_x=False)
        finally:
            _sys.stderr = _old_stderr
        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        else:
            import torch as _t
            preds = _t.stack(preds).numpy() if isinstance(preds, list) else preds.numpy()
        # preds shape: (1, horizon, 7) — values are log-price, back-transform with exp()
        p = preds[0] if preds.ndim == 3 else preds
        last_price = float(prices.iloc[-1])
        def _back(q_col):
            vals = np.exp(p[:, q_col])
            # Sanity-check: clip to ±60% of last price to guard against runaway predictions
            return np.clip(vals, last_price * 0.40, last_price * 1.60).tolist()
        future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
        return {
            "forecast":      _back(3),   # q50
            "ci_80_lo":      _back(1),   # q10
            "ci_80_hi":      _back(5),   # q90
            "ci_95_lo":      _back(0),   # q05
            "ci_95_hi":      _back(6),   # q95
            "backtest_mape": None,
            "future_dates":  [str(d.date()) for d in future_dates],
        }
    except Exception as exc:
        print(f"  [warn] TFT forecast failed: {exc}", flush=True)
        return {}


class _NBEATSBlock(_nn.Module if TORCH_OK else object):
    """Single N-BEATS block: FC stack -> backcast + forecast via basis expansion."""
    def __init__(self, input_size, theta_size, horizon, n_layers=4, hidden=256):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [_nn.Linear(input_size if i == 0 else hidden, hidden), _nn.ReLU()]
        self.fc      = _nn.Sequential(*layers)
        self.theta_b = _nn.Linear(hidden, theta_size)
        self.theta_f = _nn.Linear(hidden, theta_size)
        self.horizon = horizon
        self.B_back  = _nn.Linear(theta_size, input_size, bias=False)
        self.B_fore  = _nn.Linear(theta_size, horizon,    bias=False)

    def forward(self, x):
        h = self.fc(x)
        return self.B_back(self.theta_b(h)), self.B_fore(self.theta_f(h))


class _NHiTSBlock(_nn.Module if TORCH_OK else object):
    """Single N-HiTS block with avg-pool downsampling and linear interpolation."""
    def __init__(self, input_size, horizon, pool_size, n_layers=4, hidden=256):
        super().__init__()
        import math
        self.pool_size = pool_size
        pooled = max(1, input_size // pool_size)
        layers = []
        for i in range(n_layers):
            layers += [_nn.Linear(pooled if i == 0 else hidden, hidden), _nn.ReLU()]
        self.fc      = _nn.Sequential(*layers)
        self.back_fc = _nn.Linear(hidden, pooled)
        self.fore_fc = _nn.Linear(hidden, max(1, horizon // pool_size) + 1)
        self.horizon  = horizon
        self.pooled   = pooled

    def forward(self, x):
        import torch as _t
        pooled   = _t.nn.functional.avg_pool1d(
            x.unsqueeze(1), self.pool_size, self.pool_size, padding=0
        ).squeeze(1)
        if pooled.shape[-1] != self.pooled:
            pooled = _t.nn.functional.adaptive_avg_pool1d(
                x.unsqueeze(1), self.pooled
            ).squeeze(1)
        h        = self.fc(pooled)
        b_down   = self.back_fc(h)
        f_down   = self.fore_fc(h)
        backcast = _t.nn.functional.interpolate(
            b_down.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False
        ).squeeze(1)
        forecast = _t.nn.functional.interpolate(
            f_down.unsqueeze(1), size=self.horizon, mode="linear", align_corners=False
        ).squeeze(1)
        return backcast, forecast


def _fit_basis_model(model_name: str, prices: pd.Series, horizon: int,
                     input_size: int = 0, epochs: int = 150) -> dict:
    """
    Shared training loop for native N-BEATS and N-HiTS.
    Sliding-window dataset on log-price, window-normalised, walk-forward MAPE.
    """
    if not TORCH_OK:
        return {}
    try:
        import torch as _t
        lp = np.log(prices.values.astype(float))
        n  = len(lp)
        if input_size <= 0:
            input_size = min(5 * horizon, 252)
        if n < input_size + horizon + 20:
            return {}

        X_list, y_list = [], []
        for i in range(input_size, n - horizon + 1):
            X_list.append(lp[i - input_size:i])
            y_list.append(lp[i:i + horizon])
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        mu  = X_arr.mean(axis=1, keepdims=True)
        sig = X_arr.std(axis=1, keepdims=True) + 1e-6
        Xn  = (X_arr - mu) / sig
        yn  = (y_arr - mu) / sig

        theta = min(input_size // 4, 32)
        if model_name == "N-BEATS":
            blocks = _nn.ModuleList([
                _NBEATSBlock(input_size, theta, horizon, n_layers=4, hidden=128)
                for _ in range(3)
            ])
        else:  # N-HiTS
            pool_sizes = [1, 2, 4]
            blocks = _nn.ModuleList([
                _NHiTSBlock(input_size, horizon, ps, n_layers=4, hidden=128)
                for ps in pool_sizes
            ])

        class _Stack(_nn.Module):
            def __init__(self, blks):
                super().__init__()
                self.blks = blks
            def forward(self, x):
                res  = x
                fore = _t.zeros(x.shape[0], horizon, device=x.device)
                for blk in self.blks:
                    b, f = blk(res)
                    res  = res - b
                    fore = fore + f
                return fore

        net   = _Stack(blocks)
        opt   = _t.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = _t.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xt = _t.tensor(Xn, dtype=_t.float32)
        yt = _t.tensor(yn, dtype=_t.float32)
        split    = max(int(len(Xt) * 0.8), 10)
        Xtr, ytr = Xt[:split], yt[:split]
        Xvl, yvl = Xt[split:], yt[split:]

        best_loss, patience, best_state = np.inf, 15, None
        for ep in range(epochs):
            net.train()
            idx = _t.randperm(len(Xtr))
            for b in range(0, len(Xtr), 64):
                i = idx[b:b+64]
                opt.zero_grad()
                loss = _t.nn.functional.mse_loss(net(Xtr[i]), ytr[i])
                loss.backward()
                _t.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            if len(Xvl) > 0:
                net.eval()
                with _t.no_grad():
                    vl = _t.nn.functional.mse_loss(net(Xvl), yvl).item()
                if vl < best_loss - 1e-5:
                    best_loss, patience = vl, 15
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience -= 1
                    if patience == 0:
                        break

        if best_state:
            net.load_state_dict(best_state)
        net.eval()

        last_win = lp[-input_size:]
        mu_f  = float(last_win.mean())
        sig_f = float(last_win.std()) + 1e-6
        xf    = _t.tensor([(last_win - mu_f) / sig_f], dtype=_t.float32)
        with _t.no_grad():
            pred_norm = net(xf).numpy()[0]
        pred_lp    = pred_norm * sig_f + mu_f
        fc = np.exp(pred_lp)

        # CI from val residuals
        if len(Xvl) > 0:
            with _t.no_grad():
                vl_pred = net(Xvl).numpy()
            # Denormalise using the raw log-price windows (X_arr), not Xn
            mu_v  = X_arr[split:].mean(axis=1, keepdims=True)
            sig_v = X_arr[split:].std(axis=1, keepdims=True) + 1e-6
            vl_lp   = vl_pred * sig_v + mu_v   # predicted log-prices
            # y_arr[split:] == yn[split:] * sig_v + mu_v  (raw log-price targets)
            resid_std = max(float(np.std(vl_lp - y_arr[split:])), 1e-4)
            # backtest MAPE on last-day prediction
            val_actual = np.exp(y_arr[split:, -1])
            val_pred_e = np.exp(vl_lp[:, -1])
            valid = [(a, p) for a, p in zip(val_actual, val_pred_e) if a > 0]
            bt_mape = float(np.mean([abs(a-p)/a*100 for a, p in valid])) if valid else None
        else:
            resid_std = float(np.std(lp) * 0.02)
            bt_mape   = None

        t_arr = np.sqrt(np.arange(1, horizon + 1))
        print(f"  {model_name} MAPE: {bt_mape:.2f}%" if bt_mape else
              f"  {model_name} MAPE: n/a", flush=True)

        future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
        return {
            "forecast":      fc.tolist(),
            "ci_80_lo":      np.exp(pred_lp - 1.282 * resid_std * t_arr).tolist(),
            "ci_80_hi":      np.exp(pred_lp + 1.282 * resid_std * t_arr).tolist(),
            "ci_95_lo":      np.exp(pred_lp - 1.960 * resid_std * t_arr).tolist(),
            "ci_95_hi":      np.exp(pred_lp + 1.960 * resid_std * t_arr).tolist(),
            "backtest_mape": bt_mape,
            "future_dates":  [str(d.date()) for d in future_dates],
        }
    except Exception as exc:
        print(f"  [warn] {model_name} failed: {exc}", flush=True)
        return {}


def fit_nhits_nbeats_forecast(prices: pd.Series, horizon: int) -> tuple[dict, dict]:
    """
    Native PyTorch N-BEATS and N-HiTS (no neuralforecast/ray dependency).
    Runs whenever TORCH_OK is True.
    Returns (nhits_result, nbeats_result).
    """
    if not TORCH_OK:
        return {}, {}
    nhits_r  = _fit_basis_model("N-HiTS",  prices, horizon)
    nbeats_r = _fit_basis_model("N-BEATS", prices, horizon)
    return nhits_r, nbeats_r

def ensemble_forecast(**named_forecasts) -> dict:
    """
    Inverse-MAPE weighted blend of any number of model forecasts.
    Pass model results as keyword arguments, e.g.:
      ensemble_forecast(arima=fc_arima, ets=fc_ets, xgb=fc_xgb, prophet=fc_prophet)
    Models without a backtest_mape get a neutral weight (mape=10.0).
    """
    components = []
    for name, fc in named_forecasts.items():
        if fc and fc.get("forecast"):
            mape = fc.get("backtest_mape") or 10.0
            w    = 1.0 / max(mape, 0.1)
            components.append((name, fc, w))

    if not components:
        return {}

    total_w = sum(c[2] for c in components)
    n = min(len(c["forecast"]) for _, c, _ in components)

    def _blend(key):
        try:
            return [
                sum(c[key][i] * w
                    for _, c, w in components
                    if c.get(key) and i < len(c[key])) / total_w
                for i in range(n)
            ]
        except Exception:
            return None

    return {
        "forecast":  _blend("forecast"),
        "ci_80_lo":  _blend("ci_80_lo"),
        "ci_80_hi":  _blend("ci_80_hi"),
        "ci_95_lo":  _blend("ci_95_lo"),
        "ci_95_hi":  _blend("ci_95_hi"),
        "weights":   {name: round(w / total_w, 3) for name, _, w in components},
        "backtest_mape": None,
        "future_dates": components[0][1].get("future_dates"),
    }


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


def fetch_iv_data(ticker: str, prices: pd.Series, horizon: int) -> dict:
    """
    Fetch implied volatility from near-term options and compute an IV cone
    for the forecast horizon.
    """
    try:
        import datetime as _dt
        t       = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {}
        target_date = _dt.date.today() + _dt.timedelta(days=max(horizon // 2, 7))
        expiry = min(
            expiries,
            key=lambda d: abs((_dt.date.fromisoformat(d) - target_date).days)
        )
        # Reject if no expiry is within 3× horizon calendar days
        best_diff = abs((_dt.date.fromisoformat(expiry) - target_date).days)
        if best_diff > horizon * 3:
            return {}
        chain = t.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        last_price  = float(prices.iloc[-1])
        atm_call = calls.iloc[(calls["strike"] - last_price).abs().argsort()[:1]]
        atm_put  = puts.iloc[(puts["strike"]  - last_price).abs().argsort()[:1]]
        iv_call  = float(atm_call["impliedVolatility"].iloc[0]) if not atm_call.empty else None
        iv_put   = float(atm_put["impliedVolatility"].iloc[0])  if not atm_put.empty else None
        iv_vals  = [v for v in [iv_call, iv_put] if v is not None and 0 < v < 5]
        if not iv_vals:
            return {}
        iv = float(np.mean(iv_vals))
        t_steps  = np.arange(1, horizon + 1) / 252
        z80, z95 = 1.282, 1.960
        return {
            "iv_annualised": round(iv * 100, 1),
            "expiry":        expiry,
            "iv_cone_lo80":  (last_price * np.exp(-iv * np.sqrt(t_steps) * z80)).tolist(),
            "iv_cone_hi80":  (last_price * np.exp(+iv * np.sqrt(t_steps) * z80)).tolist(),
            "iv_cone_lo95":  (last_price * np.exp(-iv * np.sqrt(t_steps) * z95)).tolist(),
            "iv_cone_hi95":  (last_price * np.exp(+iv * np.sqrt(t_steps) * z95)).tolist(),
        }
    except Exception:
        return {}


def fetch_macro_features(prices: pd.Series, sector: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch VIX, 10Y yield, sector ETF, and SPY; align to prices.index.
    Cached at module level by (sector, period).
    """
    global _MACRO_CACHE
    cache_key = (sector, period)
    if cache_key in _MACRO_CACHE:
        return _MACRO_CACHE[cache_key].reindex(prices.index, method="ffill")
    sector_etf = _SECTOR_ETF_MAP.get(sector, "SPY")
    tickers_to_fetch = ["^VIX", "^TNX", sector_etf]
    if sector_etf != "SPY":
        tickers_to_fetch.append("SPY")
    raw = yf.download(tickers_to_fetch, period=period, auto_adjust=True,
                      progress=False)
    if raw.empty:
        return pd.DataFrame(index=prices.index, columns=MACRO_FEATURE_COLS)
    # Handle both MultiIndex and flat columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
    else:
        close = raw
    vix = close.get("^VIX", pd.Series(dtype=float))
    tnx = close.get("^TNX", pd.Series(dtype=float))
    sec = close.get(sector_etf, close.get("SPY", pd.Series(dtype=float)))
    spy = close.get("SPY", sec)
    macro = pd.DataFrame(index=raw.index)
    macro["vix_level"]      = vix
    macro["vix_5d_change"]  = vix.pct_change(5)
    macro["yield_10y"]      = tnx
    macro["sector_rel_21d"] = sec.pct_change(21).sub(spy.pct_change(21))
    macro["market_rel_21d"] = spy.pct_change(21)
    _MACRO_CACHE[cache_key] = macro
    return macro.reindex(prices.index, method="ffill")


def fetch_fundamentals(ticker: str) -> dict | None:
    """
    Pull key fundamental data from yfinance and build the `d` dict accepted
    by valuation_models.py.  Dollar amounts stored in millions; per-share
    figures (eps, bvps, dividends) kept as-is.  Returns None on failure.
    """
    print(f"  Fetching fundamental data for {ticker}…", flush=True)
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        return None

    def _m(key):
        """Return info[key] / 1e6 or 0.0 (for million-scale financials)."""
        v = info.get(key)
        return float(v) / 1e6 if v else 0.0

    def _f(key, default=None):
        v = info.get(key)
        return float(v) if v is not None else default

    price      = _f("currentPrice") or _f("regularMarketPrice") or 0.0
    market_cap = _f("marketCap") or 0.0
    shares     = _m("sharesOutstanding")           # millions
    eps        = _f("trailingEps",  0.0)
    fwd_eps    = _f("forwardEps")                  # None if missing
    revenue    = _m("totalRevenue")
    ebitda     = _m("ebitda")
    _fcf_raw   = _m("freeCashflow")
    total_debt = _m("totalDebt")
    cash       = _m("totalCash")
    # FCF sanity: yfinance sometimes misclassifies financing inflows as FCF.
    # Discard FCF when it exceeds revenue by 3× — it's not operational cash flow.
    fcf = _fcf_raw if (revenue <= 0 or _fcf_raw <= 0 or _fcf_raw <= revenue * 3) else 0.0
    bvps       = _f("bookValue",    0.0)
    beta       = _f("beta",         1.0) or 1.0
    dividends  = _f("dividendRate", 0.0) or 0.0   # annual per share
    pe_ttm     = _f("trailingPE",   0.0) or 0.0
    peg_ratio  = _f("trailingPegRatio") or _f("pegRatio") or 0.0
    sector     = info.get("sector", "Unknown")

    # Growth: prefer earnings growth, fall back to revenue growth, then 10 %
    # Cap at ±50%: yfinance can return nonsense % when losses flip sign
    _eg_raw = _f("earningsGrowth") or _f("revenueGrowth") or 0.10
    est_growth = max(-0.50, min(float(_eg_raw), 0.50))
    rev_growth = max(-0.50, min(_f("revenueGrowth") or 0.05, 0.50))

    # Analyst price target
    analyst_target = _f("targetMeanPrice")

    # ROIC & stockholders equity — try balance sheet
    roic_val          = None
    stockholders_eq   = None
    try:
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            for k in ("Stockholders Equity", "StockholdersEquity",
                      "Total Stockholder Equity", "CommonStockEquity"):
                if k in bs.index:
                    v = bs.loc[k].iloc[0]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        stockholders_eq = float(v) / 1e6
                        break
    except Exception:
        pass

    # WACC inputs — try annual income statement for interest expense & taxes
    int_exp  = None
    tax_exp  = None
    pretax   = None
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            for k in ("Interest Expense", "InterestExpense", "Net Interest Income"):
                if k in fin.index:
                    v = fin.loc[k].iloc[0]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        int_exp = abs(float(v))
                        break
            for k in ("Tax Provision", "Income Tax Expense", "IncomeTaxExpense"):
                if k in fin.index:
                    v = fin.loc[k].iloc[0]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        tax_exp = abs(float(v))
                        break
            for k in ("Pretax Income", "PretaxIncome", "Income Before Tax"):
                if k in fin.index:
                    v = fin.loc[k].iloc[0]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        pretax = float(v)
                        break
    except Exception:
        pass

    # Approximate ROIC from operating income when balance sheet data is available
    if stockholders_eq is not None:
        ic = total_debt + stockholders_eq - cash
        op_inc = _m("operatingIncome") or (ebitda * 0.7 if ebitda else 0)
        if ic > 0 and op_inc > 0:
            roic_val = op_inc * (1 - 0.21) / ic

    fcf_per_share = fcf / shares if shares > 0 else 0.0

    d = {
        "price":        price,
        "market_cap":   market_cap,
        "shares":       shares,
        "eps":          eps,
        "fwd_eps":      fwd_eps,
        "revenue":      revenue,
        "ebitda":       ebitda,
        "fcf":          fcf,
        "fcf_per_share": fcf_per_share,
        "total_debt":   total_debt,
        "cash":         cash,
        "est_growth":   est_growth,
        "rev_growth":   rev_growth,
        "beta":         beta,
        "pe_ttm":       pe_ttm,
        "peg":          peg_ratio,
        "sector":       sector,
        # wacc_raw — used by calculate_wacc()
        "wacc_raw": {
            "beta_yf":            beta,
            "total_debt_yf":      total_debt,
            "interest_expense":   int_exp,
            "income_tax_expense": tax_exp,
            "pretax_income":      pretax,
            "book_value_ps":      bvps,
        },
        # ext — used by RIM, DDM, ROIC models
        "ext": {
            "book_value_ps":       bvps,
            "dividends_per_share": dividends,
            "roic":                roic_val,
            "stockholders_equity": stockholders_eq,
        },
        # analyst consensus — displayed but not a model output
        "_analyst_target": analyst_target,
    }
    return d


def run_fundamental_valuations(d: dict) -> dict:
    """
    Run all applicable valuation models from valuation_models.py against the
    fundamental data dict `d`.  Uses sector-level benchmark multiples when live
    peer data is unavailable.

    Returns a dict with keys:
      "models"    — list of {method, fair_value, mos_value, reliable, note}
      "valid_fvs" — list of valid fair_value floats (for composite calc)
      "composite" — median of valid_fvs, or None
      "low"       — 25th-percentile of valid_fvs
      "high"      — 75th-percentile of valid_fvs
      "analyst_target" — analyst consensus from info (may be None)
    """
    try:
        from valuation_models import (
            run_dcf, run_three_stage_dcf,
            run_pe, run_pfcf, run_ev_ebitda,
            run_graham_number,
            run_rim, run_ddm_hmodel, run_roic_excess_return,
        )
    except ImportError as exc:
        print(f"  [warn] valuation_models import failed: {exc}", flush=True)
        return {}

    sector = d.get("sector", "Unknown")
    sm     = _SECTOR_BM.get(sector, _DEFAULT_BM)
    bm     = {**sm, "sector_name": sector}

    rows   = []

    def _run(name, fn, *args):
        try:
            r = fn(*args)
            if r is None:
                return
            fv = r.get("fair_value") if isinstance(r, dict) else r
            if fv is None or fv <= 0:
                return
            mos = (r.get("mos_value") if isinstance(r, dict) else round(fv * 0.80, 2)) or 0.0
            rel = r.get("reliable", True) if isinstance(r, dict) else True
            warn = r.get("warning") if isinstance(r, dict) else None
            rows.append({
                "method":     name,
                "fair_value": round(float(fv), 2),
                "mos_value":  round(float(mos), 2),
                "reliable":   rel,
                "note":       warn or "",
            })
        except Exception:
            pass

    _run("DCF (2-Stage)",       run_dcf,              d)
    _run("DCF (3-Stage)",       run_three_stage_dcf,  d)
    _run("P/E",                 run_pe,               d, bm)
    _run("P/FCF",               run_pfcf,             d, bm)
    _run("EV/EBITDA",           run_ev_ebitda,        d, bm)
    _run("Graham Number",       run_graham_number,    d)
    _run("Residual Income",     run_rim,              d)
    _run("DDM (H-Model)",       run_ddm_hmodel,       d)
    _run("ROIC Excess Return",  run_roic_excess_return, d)

    valid_fvs = [r["fair_value"] for r in rows if r["reliable"] and r["fair_value"] > 0]

    composite = None
    low = high = None
    if valid_fvs:
        sv = sorted(valid_fvs)
        n  = len(sv)
        composite = round(sv[n // 2] if n % 2 == 1 else (sv[n//2-1]+sv[n//2])/2, 2)
        q1_idx = max(0, n // 4)
        q3_idx = min(n - 1, (n * 3) // 4)
        low  = round(sv[q1_idx], 2)
        high = round(sv[q3_idx], 2)

    print(f"  Fundamental models: {len(rows)} ran  |  "
          f"composite target: {composite or 'n/a'}", flush=True)

    return {
        "models":          rows,
        "valid_fvs":       valid_fvs,
        "composite":       composite,
        "low":             low,
        "high":            high,
        "analyst_target":  d.get("_analyst_target"),
    }


def fetch_estimate_revisions(ticker: str) -> dict:
    """
    Pull analyst EPS and revenue estimate revision trends from yfinance.
    Returns direction (improving/deteriorating/stable) and % change vs 90 days ago.
    """
    try:
        t  = yf.Ticker(ticker)
        ee = t.earnings_estimate
        re = t.revenue_estimate
        def _parse_revisions(df, label):
            if df is None or df.empty:
                return {}
            row = df.loc["0q"] if "0q" in df.index else df.iloc[0]
            avg = float(row.get("avg", 0) or 0)
            ago90 = float(row.get("90daysAgo", 0) or 0)
            ago30 = float(row.get("30daysAgo", 0) or 0)
            if ago90 == 0:
                return {"direction": "stable", "magnitude": None,
                        "current": avg, "label": label}
            magnitude = (avg - ago90) / abs(ago90) * 100
            if magnitude > 1:
                direction = "improving"
            elif magnitude < -1:
                direction = "deteriorating"
            else:
                direction = "stable"
            return {
                "direction": direction,
                "magnitude": round(magnitude, 1),
                "current":   round(avg, 2),
                "ago30":     round(ago30, 2),
                "ago90":     round(ago90, 2),
                "label":     label,
            }
        eps_rev = _parse_revisions(ee, "EPS")
        rev_rev = _parse_revisions(re, "Revenue")
        if not eps_rev and not rev_rev:
            return {}
        parts = []
        for r in [eps_rev, rev_rev]:
            if r and r.get("magnitude") is not None:
                parts.append(f"{r['label']} estimates {r['direction']} "
                             f"{r['magnitude']:+.1f}% vs 90d ago")
        return {
            "eps":     eps_rev,
            "revenue": rev_rev,
            "summary": "; ".join(parts) if parts else "No revision data available.",
        }
    except Exception:
        return {}


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


def detect_regime(prices: pd.Series) -> dict:
    """
    Classify current market regime from price structure.
    Returns Bull / Bear / Neutral based on 50/200-day MAs.
    """
    if len(prices) < 200:
        return {"regime": "Neutral", "note": "Insufficient history for regime detection."}
    ma50  = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    current   = float(prices.iloc[-1])
    ma50_now  = float(ma50.iloc[-1])
    ma200_now = float(ma200.iloc[-1])
    slope200  = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20] if ma200.iloc[-20] else 0
    above200  = current > ma200_now
    slope_up  = slope200 > 0
    if above200 and slope_up:
        regime = "Bull"
    elif not above200 and not slope_up:
        regime = "Bear"
    else:
        regime = "Neutral"
    # Golden / death cross detection (within last 5 sessions)
    cross = None
    try:
        recent_50  = ma50.iloc[-5:]
        recent_200 = ma200.iloc[-5:]
        if (recent_50.iloc[-1] > recent_200.iloc[-1]) and (recent_50.iloc[0] <= recent_200.iloc[0]):
            cross = "golden"
        elif (recent_50.iloc[-1] < recent_200.iloc[-1]) and (recent_50.iloc[0] >= recent_200.iloc[0]):
            cross = "death"
    except Exception:
        pass
    note_parts = [
        f"Current regime: {regime}.",
        f"Price is {'above' if above200 else 'below'} the 200-day MA (${ma200_now:,.2f}).",
        f"200-day MA slope: {slope200*100:+.2f}% over 20 sessions.",
    ]
    if cross == "golden":
        note_parts.append("Golden cross detected in the last 5 sessions (50-day MA crossed above 200-day MA).")
    elif cross == "death":
        note_parts.append("Death cross detected in the last 5 sessions (50-day MA crossed below 200-day MA).")
    return {
        "regime":       regime,
        "ma50":         round(ma50_now, 2),
        "ma200":        round(ma200_now, 2),
        "slope200_pct": round(slope200 * 100, 3),
        "golden_cross": cross == "golden",
        "death_cross":  cross == "death",
        "note":         " ".join(note_parts),
    }


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
    all_prices     = prices.values

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

    def sharpe(pred_list):
        pred   = np.array(pred_list)
        actual = np.array(all_actual)
        mask   = ~np.isnan(pred) & (np.abs(actual) > 1e-6)
        if mask.sum() < horizon * 2:
            return None
        p, a = pred[mask], actual[mask]
        n_full = (len(p) // horizon) * horizon
        if n_full < horizon * 2:
            return None
        daily_rets = []
        for i in range(0, n_full - horizon, horizon):
            sig = 1 if p[i + horizon - 1] > a[i] * 1.01 else 0
            idx_start = len(all_actual) - len(pred) + i
            idx_end   = min(idx_start + horizon + 1, len(all_actual))
            block_px  = np.array(all_prices[idx_start:idx_end])
            if len(block_px) < 2:
                continue
            block_rets = np.diff(np.log(block_px + 1e-10))
            daily_rets.extend((block_rets * sig).tolist())
        if len(daily_rets) < 5:
            return None
        dr = np.array(daily_rets)
        ann_vol = np.std(dr, ddof=1) * np.sqrt(252)
        if ann_vol < 1e-8:
            return None
        return float(np.mean(dr) * 252 / ann_vol)

    arima_m  = metrics(all_arima_pred) if do_arima else {}
    ets_m    = metrics(all_ets_pred)   if do_ets   else {}
    naive_m  = metrics(all_naive_pred)
    if do_arima and arima_m:
        arima_m["sharpe"] = sharpe(all_arima_pred)
    if do_ets and ets_m:
        ets_m["sharpe"]   = sharpe(all_ets_pred)
    naive_m["sharpe"] = sharpe(all_naive_pred)

    return {
        "dates":          all_dates,
        "actual":         all_actual,
        "arima_preds":    all_arima_pred,
        "ets_preds":      all_ets_pred,
        "naive_preds":    all_naive_pred,
        "arima_metrics":  arima_m,
        "ets_metrics":    ets_m,
        "naive_metrics":  naive_m,
    }


def compute_multihorizon(all_forecasts: dict, horizon: int) -> list:
    """
    Extract model endpoint values at milestone trading days within the forecast horizon.
    all_forecasts: dict of model_name → forecast list (prices)
    Returns list of {day, values, median, q25, q75, min, max}
    """
    milestones = [d for d in [7, 14, 21, 30, 45, 63, 90, 126, 252] if d <= horizon]
    rows = []
    for day in milestones:
        idx = day - 1   # 0-indexed
        vals = {}
        for name, fc in all_forecasts.items():
            if fc and len(fc) > idx:
                vals[name] = float(fc[idx])
        if not vals:
            continue
        v = sorted(vals.values())
        n = len(v)
        rows.append({
            "day":    day,
            "values": vals,
            "median": round(v[n // 2], 4),
            "q25":    round(v[max(0, n // 4)], 4),
            "q75":    round(v[min(n - 1, (n * 3) // 4)], 4),
            "min":    round(v[0], 4),
            "max":    round(v[-1], 4),
        })
    return rows


def compute_stl(prices: pd.Series) -> dict:
    """
    STL (Seasonal-Trend decomposition using LOESS) on log-price.
    Period = 252 (annual trading-day cycle). Requires ≥2 full periods.
    Returns dict with arrays for trend, seasonal, residual, dates.
    """
    try:
        from statsmodels.tsa.seasonal import STL as _STL
        lp = np.log(prices.values.astype(float))
        if len(lp) < 504:  # need ≥2 years for meaningful decomposition
            return {}
        result = _STL(lp, period=252, robust=True).fit()
        dates = [str(d.date()) for d in prices.index]
        # Back-transform trend to price space for display
        return {
            "dates":    dates,
            "trend":    np.exp(result.trend).tolist(),
            "seasonal": result.seasonal.tolist(),   # keep in log-return scale
            "residual": result.resid.tolist(),
            "strength_trend":   round(float(max(0, 1 - np.var(result.resid) /
                                                np.var(result.trend + result.resid))), 3),
            "strength_seasonal": round(float(max(0, 1 - np.var(result.resid) /
                                                 np.var(result.seasonal + result.resid))), 3),
        }
    except Exception:
        return {}


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
    xgb_result: dict = None,
    do_ensemble: bool = False,
    prophet_result: dict = None,
    lstm_result:    dict = None,
    timesfm_result: dict = None,
    chronos_result: dict = None,
    fundamental_results: dict = None,
    montecarlo_result:   dict = None,
    iv_result:           dict = None,
    tft_result:          dict = None,
    regime_result:       dict = None,
    estimate_revisions:  dict = None,
    nhits_result:        dict = None,
    nbeats_result:       dict = None,
    stl_result:          dict = None,
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

    # XGBoost forecast
    xgb_fc_px = xgb_fc_lo80 = xgb_fc_hi80 = xgb_fc_lo95 = xgb_fc_hi95 = []
    xgb_bt_mape = None
    if xgb_result and xgb_result.get("forecast"):
        xgb_fc_px   = xgb_result["forecast"]
        xgb_fc_lo80 = xgb_result.get("ci_80_lo", [])
        xgb_fc_hi80 = xgb_result.get("ci_80_hi", [])
        xgb_fc_lo95 = xgb_result.get("ci_95_lo", [])
        xgb_fc_hi95 = xgb_result.get("ci_95_hi", [])
        xgb_bt_mape = xgb_result.get("backtest_mape")

    # Prophet forecast
    prophet_fc_px = prophet_fc_lo80 = prophet_fc_hi80 = \
    prophet_fc_lo95 = prophet_fc_hi95 = []
    prophet_bt_mape = None
    if prophet_result and prophet_result.get("forecast"):
        prophet_fc_px   = prophet_result["forecast"]
        prophet_fc_lo80 = prophet_result.get("ci_80_lo", [])
        prophet_fc_hi80 = prophet_result.get("ci_80_hi", [])
        prophet_fc_lo95 = prophet_result.get("ci_95_lo", [])
        prophet_fc_hi95 = prophet_result.get("ci_95_hi", [])
        prophet_bt_mape = prophet_result.get("backtest_mape")

    # LSTM forecast
    lstm_fc_px = lstm_fc_lo80 = lstm_fc_hi80 = \
    lstm_fc_lo95 = lstm_fc_hi95 = []
    lstm_bt_mape = None
    if lstm_result and lstm_result.get("forecast"):
        lstm_fc_px   = lstm_result["forecast"]
        lstm_fc_lo80 = lstm_result.get("ci_80_lo", [])
        lstm_fc_hi80 = lstm_result.get("ci_80_hi", [])
        lstm_fc_lo95 = lstm_result.get("ci_95_lo", [])
        lstm_fc_hi95 = lstm_result.get("ci_95_hi", [])
        lstm_bt_mape = lstm_result.get("backtest_mape")

    # TimesFM forecast
    timesfm_fc_px = timesfm_fc_lo80 = timesfm_fc_hi80 = \
    timesfm_fc_lo95 = timesfm_fc_hi95 = []
    if timesfm_result and timesfm_result.get("forecast"):
        timesfm_fc_px   = timesfm_result["forecast"]
        timesfm_fc_lo80 = timesfm_result.get("ci_80_lo", [])
        timesfm_fc_hi80 = timesfm_result.get("ci_80_hi", [])
        timesfm_fc_lo95 = timesfm_result.get("ci_95_lo", [])
        timesfm_fc_hi95 = timesfm_result.get("ci_95_hi", [])

    # Chronos forecast
    chronos_fc_px = chronos_fc_lo80 = chronos_fc_hi80 = \
    chronos_fc_lo95 = chronos_fc_hi95 = []
    if chronos_result and chronos_result.get("forecast"):
        chronos_fc_px   = chronos_result["forecast"]
        chronos_fc_lo80 = chronos_result.get("ci_80_lo", [])
        chronos_fc_hi80 = chronos_result.get("ci_80_hi", [])
        chronos_fc_lo95 = chronos_result.get("ci_95_lo", [])
        chronos_fc_hi95 = chronos_result.get("ci_95_hi", [])

    # Monte Carlo GBM forecast
    mc_fc_px = mc_fc_lo80 = mc_fc_hi80 = mc_fc_lo95 = mc_fc_hi95 = []
    mc_end = mc_chg = None
    if montecarlo_result and montecarlo_result.get("forecast"):
        mc_fc_px   = montecarlo_result["forecast"]
        mc_fc_lo80 = montecarlo_result.get("ci_80_lo", [])
        mc_fc_hi80 = montecarlo_result.get("ci_80_hi", [])
        mc_fc_lo95 = montecarlo_result.get("ci_95_lo", [])
        mc_fc_hi95 = montecarlo_result.get("ci_95_hi", [])

    # TFT forecast
    tft_fc_px = tft_fc_lo80 = tft_fc_hi80 = tft_fc_lo95 = tft_fc_hi95 = []
    tft_end = tft_chg = None
    if tft_result and tft_result.get("forecast"):
        tft_fc_px   = tft_result["forecast"]
        tft_fc_lo80 = tft_result.get("ci_80_lo", [])
        tft_fc_hi80 = tft_result.get("ci_80_hi", [])
        tft_fc_lo95 = tft_result.get("ci_95_lo", [])
        tft_fc_hi95 = tft_result.get("ci_95_hi", [])

    # N-HiTS forecast
    nhits_fc_px = nhits_fc_lo80 = nhits_fc_hi80 = nhits_fc_lo95 = nhits_fc_hi95 = []
    nhits_end = nhits_chg = None
    if nhits_result and nhits_result.get("forecast"):
        nhits_fc_px   = nhits_result["forecast"]
        nhits_fc_lo80 = nhits_result.get("ci_80_lo", [])
        nhits_fc_hi80 = nhits_result.get("ci_80_hi", [])
        nhits_fc_lo95 = nhits_result.get("ci_95_lo", [])
        nhits_fc_hi95 = nhits_result.get("ci_95_hi", [])

    # N-BEATS forecast
    nbeats_fc_px = nbeats_fc_lo80 = nbeats_fc_hi80 = nbeats_fc_lo95 = nbeats_fc_hi95 = []
    nbeats_end = nbeats_chg = None
    if nbeats_result and nbeats_result.get("forecast"):
        nbeats_fc_px   = nbeats_result["forecast"]
        nbeats_fc_lo80 = nbeats_result.get("ci_80_lo", [])
        nbeats_fc_hi80 = nbeats_result.get("ci_80_hi", [])
        nbeats_fc_lo95 = nbeats_result.get("ci_95_lo", [])
        nbeats_fc_hi95 = nbeats_result.get("ci_95_hi", [])

    # Ensemble forecast (inverse-MAPE weighted blend)
    ens_fc_px = ens_fc_lo80 = ens_fc_hi80 = ens_fc_lo95 = ens_fc_hi95 = []
    ens_weights = {}
    if do_ensemble:
        _arima_mape = (bt or {}).get("arima_metrics", {}).get("mape")
        _ets_mape   = (bt or {}).get("ets_metrics",   {}).get("mape")
        _ens = ensemble_forecast(
            arima={"forecast": arima_fc_px, "ci_80_lo": arima_fc_lo80,
                   "ci_80_hi": arima_fc_hi80, "ci_95_lo": arima_fc_lo95,
                   "ci_95_hi": arima_fc_hi95, "backtest_mape": _arima_mape}
                  if arima_fc_px else {},
            ets=  {"forecast": ets_fc_px,   "ci_80_lo": ets_fc_lo80,
                   "ci_80_hi": ets_fc_hi80,   "ci_95_lo": ets_fc_lo95,
                   "ci_95_hi": ets_fc_hi95,   "backtest_mape": _ets_mape}
                  if ets_fc_px else {},
            xgb=  xgb_result or {},
            prophet= prophet_result or {},
            lstm=    lstm_result    or {},
            montecarlo=montecarlo_result or {},
            tft=tft_result or {},
            nhits=nhits_result or {},
            nbeats=nbeats_result or {},
        )
        ens_fc_px   = _ens.get("forecast") or []
        ens_fc_lo80 = _ens.get("ci_80_lo") or []
        ens_fc_hi80 = _ens.get("ci_80_hi") or []
        ens_fc_lo95 = _ens.get("ci_95_lo") or []
        ens_fc_hi95 = _ens.get("ci_95_hi") or []
        ens_weights = _ens.get("weights", {})

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
    xgb_end,   xgb_chg   = _summary(xgb_fc_px)
    prophet_end, prophet_chg = _summary(prophet_fc_px)
    lstm_end,    lstm_chg    = _summary(lstm_fc_px)
    timesfm_end, timesfm_chg = _summary(timesfm_fc_px)
    chronos_end, chronos_chg = _summary(chronos_fc_px)
    mc_end,    mc_chg    = _summary(mc_fc_px)
    tft_end,   tft_chg   = _summary(tft_fc_px)
    nhits_end, nhits_chg = _summary(nhits_fc_px)
    nbeats_end, nbeats_chg = _summary(nbeats_fc_px)
    ens_end,   ens_chg   = _summary(ens_fc_px)

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

    # XGB backtest metrics (MAPE only; RMSE/MAE/dir not computed in walk-forward)
    xgb_m = {}
    if xgb_bt_mape is not None:
        xgb_m = {"mape": xgb_bt_mape}

    metrics_html = ""
    if bt or xgb_m:
        rows = ""
        for label, m in [("ARIMA", arima_m), ("ETS", ets_m)]:
            if not m:
                continue
            _sh = m.get("sharpe")
            _sh_base = naive_m.get("sharpe")
            _sh_str = f"{_sh:.2f}" if _sh is not None else "—"
            _sh_better = _sh is not None and _sh_base is not None and _sh > _sh_base and _sh > 0
            _sh_clr = ACCENT_GRN if _sh_better else (ACCENT_RED if (_sh is not None and _sh < 0) else TEXT_SUB)
            rows += f"""
            <tr>
              <td><b>{label}</b></td>
              {_metric_td(m, 'rmse', naive_m)}
              {_metric_td(m, 'mae',  naive_m)}
              {_metric_td(m, 'mape', naive_m)}
              {_dir_td(m, naive_m)}
              <td style="color:{_sh_clr}">{_sh_str}</td><td></td>
            </tr>"""
        if xgb_m:
            _xgb_mape_v = xgb_m.get("mape")
            _xgb_naive  = naive_m.get("mape")
            _xmape_str  = f"{_xgb_mape_v:.2f}%" if _xgb_mape_v is not None else "—"
            if _xgb_mape_v is not None and _xgb_naive is not None:
                _xclr = ACCENT_GRN if _xgb_mape_v < _xgb_naive else ACCENT_RED
                _xdlt = _xgb_mape_v - _xgb_naive
                _xdlt_str = f'<span style="color:{_xclr}">{"+" if _xdlt>=0 else ""}{_xdlt:.2f} vs naive</span>'
            else:
                _xdlt_str = "—"
            rows += f"""
            <tr>
              <td><b>XGBoost</b></td>
              <td>—</td><td>walk-forward log-return MAPE only</td>
              <td>—</td><td></td>
              <td>{_xmape_str}</td><td>{_xdlt_str}</td>
              <td>—</td><td></td>
              <td style="color:{TEXT_SUB}">—</td><td></td>
            </tr>"""
        if ens_weights:
            _w_str = " / ".join(f"{k.upper()}:{v:.0%}" for k, v in ens_weights.items())
            rows += f"""
            <tr>
              <td><b>Ensemble</b></td>
              <td colspan="10" style="color:{TEXT_SUB}">
                Inverse-MAPE blend — weights: {_w_str}
              </td>
            </tr>"""
        if bt:
            _sh_naive = naive_m.get("sharpe")
            _sh_naive_str = f"{_sh_naive:.2f}" if _sh_naive is not None else "—"
            rows += f"""
            <tr style="color:{TEXT_SUB}">
              <td>Naive (flat)</td>
              <td>{naive_m.get('rmse', 0):.4f}</td><td>baseline</td>
              <td>{naive_m.get('mae',  0):.4f}</td><td>baseline</td>
              <td>{naive_m.get('mape', 0):.2f}%</td><td>baseline</td>
              <td>{naive_m.get('dir_acc', 50):.1f}%</td><td>baseline</td>
              <td>{_sh_naive_str}</td><td>baseline</td>
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
                <th>Sharpe</th><th></th>
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
    _help_data = {}   # key → {title, body}  — populated per card, emitted as JS later
    _card_id_counter = [0]

    def _stat_card(title, val, sub, clr=None, info=None):
        c   = clr or TEXT_MAIN
        btn = ""
        if info:
            _card_id_counter[0] += 1
            key = f"card-{_card_id_counter[0]}"
            _help_data[key] = {"title": title, "body": info}
            safe_key = key.replace("'", "\\'")
            btn = (f'<button class="help-btn" onclick="showHelp(\'{safe_key}\',event)" '
                   f'title="About this estimate" aria-label="Help">ⓘ</button>')
        return (f'<div class="stat-card">{btn}'
                f'<div class="stat-val" style="color:{c}">{val}</div>'
                f'<div class="stat-sub">{title}</div>'
                f'<div class="stat-note">{sub}</div>'
                f'</div>')

    _badge_models = " + ".join(m for m, ok in [
        ("ARIMA",        do_arima),
        ("ETS",          do_ets),
        ("XGB",          bool((xgb_result        or {}).get("forecast"))),
        ("Prophet",      bool((prophet_result     or {}).get("forecast"))),
        ("LSTM",         bool((lstm_result         or {}).get("forecast"))),
        ("TimesFM",      bool((timesfm_result      or {}).get("forecast"))),
        ("Chronos",      bool((chronos_result      or {}).get("forecast"))),
        ("Monte Carlo",  bool((montecarlo_result   or {}).get("forecast"))),
        ("TFT",          bool((tft_result          or {}).get("forecast"))),
        ("N-HiTS",       bool((nhits_result        or {}).get("forecast"))),
        ("N-BEATS",      bool((nbeats_result       or {}).get("forecast"))),
    ] if ok) or model_arg

    # ── Model disagreement: std of all model endpoint forecasts ──────────────────
    _endpoint_vals = [v for v in [
        arima_end, ets_end, xgb_end, prophet_end, lstm_end,
        timesfm_end, chronos_end, mc_end, tft_end, nhits_end, nbeats_end,
    ] if v is not None]
    _disagree_score = None
    _disagree_label = ""
    _disagree_clr   = TEXT_SUB
    if len(_endpoint_vals) >= 2:
        _disagree_cv = float(np.std(_endpoint_vals) / np.mean(_endpoint_vals) * 100)
        _disagree_score = round(_disagree_cv, 1)
        if _disagree_cv < 5:
            _disagree_label = "low"
            _disagree_clr   = ACCENT_GRN
        elif _disagree_cv < 15:
            _disagree_label = "moderate"
            _disagree_clr   = ACCENT_AMB
        else:
            _disagree_label = "high"
            _disagree_clr   = ACCENT_RED

    _regime = (regime_result or {}).get("regime", "")
    _regime_clr = {"Bull": ACCENT_GRN, "Bear": ACCENT_RED}.get(_regime, TEXT_SUB)
    _regime_badge = (f'<span class="badge" style="background:{_regime_clr}22;color:{_regime_clr}">'
                     f'regime: {_regime}</span>') if _regime else ""

    _disagree_badge = (
        f'<span class="badge" style="background:{_disagree_clr}22;color:{_disagree_clr}">'
        f'disagreement: {_disagree_score:.1f}%</span>'
    ) if _disagree_score is not None else ""

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
            info=(
                "ARIMA (AutoRegressive Integrated Moving Average) is a classical time-series model. "
                "It fits on log-returns (price changes) to make the series stationary, then captures "
                "autocorrelation structure using autoregressive (AR) and moving-average (MA) terms. "
                "The order (p,d,q) is selected automatically via AIC minimisation. "
                "Confidence bands are scaled by GARCH(1,1) conditional volatility so they widen "
                "during volatile regimes and narrow in calm ones."
            ),
        )
    if ets_end is not None:
        stat_cards += _stat_card(
            f"ETS t+{horizon}",
            f"{currency} {ets_end:,.2f}",
            _fmt_pct(ets_chg),
            _color_delta(ets_chg),
            info=(
                "ETS (Error-Trend-Seasonality) is an exponential smoothing state-space model. "
                "It fits on log-price levels (not returns) and uses a damped additive trend to avoid "
                "over-extrapolating long-run momentum. The damping parameter &phi; shrinks the trend "
                "toward zero as the horizon grows, reflecting mean-reversion uncertainty. "
                "Confidence bands are scaled by GARCH(1,1) conditional volatility."
            ),
        )
    if xgb_end is not None:
        stat_cards += _stat_card(
            f"XGBoost t+{horizon}",
            f"{currency} {xgb_end:,.2f}",
            _fmt_pct(xgb_chg),
            _color_delta(xgb_chg),
            info=(
                "XGBoost is a gradient-boosted decision tree model trained on 9 engineered price features: "
                "momentum (5/21/63-day returns), volatility (21-day realised vol), RSI, "
                "distance from 50-day and 200-day moving averages, volume ratio, and log-price. "
                "The target is the cumulative log-return over the forecast horizon. "
                "Walk-forward cross-validation estimates out-of-sample MAPE, which determines the "
                "model&rsquo;s weight in the ensemble."
            ),
        )
    if prophet_end is not None:
        stat_cards += _stat_card(
            f"Prophet t+{horizon}",
            f"{currency} {prophet_end:,.2f}",
            _fmt_pct(prophet_chg),
            _color_delta(prophet_chg),
            info=(
                "Prophet is Meta&rsquo;s open-source additive forecasting model. "
                "It decomposes the log-price series into a piecewise-linear trend, "
                "optional Fourier-series seasonality components, and a holiday effect term. "
                "The model is robust to missing data and trend shifts. "
                "Fitting is done in log-price space; results are exponentiated back to price. "
                "Uncertainty intervals use 500 posterior samples from the internal Stan model."
            ),
        )
    if lstm_end is not None:
        stat_cards += _stat_card(
            f"LSTM t+{horizon}",
            f"{currency} {lstm_end:,.2f}",
            _fmt_pct(lstm_chg),
            _color_delta(lstm_chg),
            info=(
                "LSTM (Long Short-Term Memory) is a recurrent neural network with 2 hidden layers "
                "(64 units each) trained on 20-day sliding windows of 9 price features. "
                "The network predicts the cumulative log-return over the forecast horizon. "
                "Uncertainty is estimated via MC Dropout: 200 stochastic forward passes are run at "
                "inference time with dropout active, and the standard deviation of predictions "
                "forms the confidence interval. This approximates a Bayesian posterior over weights."
            ),
        )
    if timesfm_end is not None:
        stat_cards += _stat_card(
            f"TimesFM t+{horizon}",
            f"{currency} {timesfm_end:,.2f}",
            _fmt_pct(timesfm_chg),
            _color_delta(timesfm_chg),
            info=(
                "TimesFM is Google&rsquo;s foundation time-series model (200M parameters). "
                "It was pre-trained on a large corpus of real-world time series across many domains. "
                "The model takes up to 512 historical price points as context and generates a "
                "probabilistic forecast with quantile outputs. No fine-tuning is performed; "
                "the model generalises zero-shot from its pre-training."
            ),
        )
    if chronos_end is not None:
        stat_cards += _stat_card(
            f"Chronos t+{horizon}",
            f"{currency} {chronos_end:,.2f}",
            _fmt_pct(chronos_chg),
            _color_delta(chronos_chg),
            info=(
                "Chronos is Amazon&rsquo;s foundation forecasting model based on the T5 language model "
                "architecture. It tokenises time-series values into discrete bins and treats forecasting "
                "as a language-modelling task. 200 Monte Carlo samples are drawn at inference time to "
                "produce the confidence interval. Like TimesFM, it operates zero-shot with no "
                "stock-specific fine-tuning."
            ),
        )
    if ens_end is not None:
        _w_str = " | ".join(f"{k.upper()}:{v:.0%}" for k, v in ens_weights.items())
        stat_cards += _stat_card(
            f"Ensemble t+{horizon}",
            f"{currency} {ens_end:,.2f}",
            _fmt_pct(ens_chg),
            _color_delta(ens_chg),
            info=(
                "The Ensemble blends all available model forecasts using inverse-MAPE weighting: "
                "models with lower walk-forward error receive proportionally more weight. "
                "This rewards accuracy and automatically down-weights poorly calibrated models. "
                f"Current weights &mdash; {_w_str if _w_str else 'computed at runtime'}. "
                "The ensemble forecast is generally more stable than any single model."
            ),
        )
    stat_cards += _stat_card(
        f"Naive t+{horizon}",
        f"{currency} {last_price:,.2f}",
        "flat (random walk)",
        TEXT_SUB,
        info=(
            "The Naive (random walk) forecast assumes tomorrow&rsquo;s price equals today&rsquo;s price. "
            "Under the Efficient Market Hypothesis, this is a surprisingly strong baseline for short horizons. "
            "All other models are benchmarked against it: a model that cannot beat the naive forecast "
            "on RMSE or directional accuracy has no predictive edge on this stock over this horizon."
        ),
    )
    if _disagree_score is not None:
        stat_cards += _stat_card(
            "Model Disagreement",
            f"{_disagree_score:.1f}%",
            f"{_disagree_label} · {len(_endpoint_vals)} models",
            _disagree_clr,
            info=(
                "Model Disagreement is the coefficient of variation (CV = std ÷ mean × 100%) "
                "of the endpoint price forecasts across all active models. "
                "When models strongly agree (low CV), the ensemble forecast is more reliable. "
                "High disagreement means models are seeing very different signals — the true "
                "outcome is genuinely uncertain and the forecast should be treated with caution. "
                "A CV below 5% is low, 5–15% is moderate, above 15% is high."
            ),
        )
    if mc_end is not None:
        stat_cards += _stat_card(
            f"Monte Carlo t+{horizon}",
            f"{currency} {mc_end:,.2f}",
            _fmt_pct(mc_chg),
            _color_delta(mc_chg),
            info=(
                "Monte Carlo simulation uses Geometric Brownian Motion (GBM), the mathematical "
                "model underlying the Black-Scholes option pricing formula. Daily drift (μ) and "
                "volatility (σ) are estimated from historical log-returns. 1,000 independent "
                "price paths are simulated forward; the median path is the central forecast and "
                "the 5th/10th/90th/95th percentiles form the confidence bands. GBM assumes "
                "log-normally distributed returns with constant volatility — it cannot model "
                "regime changes or fat tails."
            ),
        )
    if tft_end is not None:
        stat_cards += _stat_card(
            f"TFT t+{horizon}",
            f"{currency} {tft_end:,.2f}",
            _fmt_pct(tft_chg),
            _color_delta(tft_chg),
            info=(
                "Temporal Fusion Transformer (TFT) is an attention-based deep learning model "
                "combining LSTMs with multi-head attention for interpretable time-series forecasting. "
                "It natively produces quantile outputs (5th–95th percentile) as confidence intervals. "
                "The model is trained on the last 500 trading days with optional macro covariates "
                "(VIX, 10Y yield, sector ETF). Architecture: hidden_size=16, 1 LSTM layer, 30 epochs. "
                "No walk-forward MAPE is computed due to training cost."
            ),
        )

    if nhits_end is not None:
        stat_cards += _stat_card(
            f"N-HiTS t+{horizon}",
            f"{currency} {nhits_end:,.2f}",
            _fmt_pct(nhits_chg),
            _color_delta(nhits_chg),
            info=(
                "N-HiTS (Neural Hierarchical Interpolation for Time Series) is a pure neural network "
                "architecture designed specifically for time series. It uses hierarchical interpolation "
                "— stacking blocks that each model different frequency components of the signal — "
                "without any recurrence or attention. Each block uses a doubly-residual stacking "
                "principle: predictions and backcast residuals are passed forward. N-HiTS is "
                "state-of-the-art on M3/M4/M5 benchmarks and consistently outperforms LSTM with "
                "lower computational cost."
            ),
        )
    if nbeats_end is not None:
        stat_cards += _stat_card(
            f"N-BEATS t+{horizon}",
            f"{currency} {nbeats_end:,.2f}",
            _fmt_pct(nbeats_chg),
            _color_delta(nbeats_chg),
            info=(
                "N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep neural network "
                "built entirely from fully-connected layers and basis function expansions. Two variants "
                "exist: generic (learns arbitrary basis functions) and interpretable (decomposes into "
                "trend and seasonality bases). Like N-HiTS, it uses no recurrence or convolution, "
                "making it faster and more interpretable than LSTM or TFT. N-BEATS won the M4 competition."
            ),
        )

    # ── Fundamental composite stat card ───────────────────────────────────────
    _fres = fundamental_results or {}
    _fcomp = _fres.get("composite")
    if _fcomp and last_price > 0:
        _fupside = (_fcomp / last_price - 1) * 100
        stat_cards += _stat_card(
            "Fundamental Target",
            f"{currency} {_fcomp:,.2f}",
            _fmt_pct(_fupside),
            _color_delta(_fupside),
            info=(
                "The Fundamental Target is the median fair value across all valuation models that "
                "produced a reliable estimate: DCF (discounted cash flow), DDM H-Model (dividend discount), "
                "Residual Income, ROIC Excess Return, Graham Number, EV/EBITDA peer multiple, P/FCF peer "
                "multiple, and P/E peer multiple. Models flagged as unreliable (e.g. negative FCF, no "
                "dividends) are excluded from the median. The IQR band (25th&ndash;75th percentile) is "
                "shown as a shaded region on the chart."
            ),
        )
    _analyst_tgt = _fres.get("analyst_target")
    if _analyst_tgt and last_price > 0:
        _atgt_chg = (_analyst_tgt / last_price - 1) * 100
        stat_cards += _stat_card(
            "Analyst Consensus",
            f"{currency} {_analyst_tgt:,.2f}",
            _fmt_pct(_atgt_chg),
            _color_delta(_atgt_chg),
            info=(
                "The Analyst Consensus target is the mean 12-month price target across all sell-side "
                "analysts covering this stock, as reported by Yahoo Finance. This is a fundamental "
                "estimate, not a statistical model forecast. It reflects analysts&rsquo; DCF and "
                "comparable-company models, management guidance, and sector views. "
                "Consensus targets tend to lag price moves and carry an upward bias."
            ),
        )
    _iv = iv_result or {}
    if _iv.get("iv_annualised") and last_price > 0:
        stat_cards += _stat_card(
            "ATM Implied Vol",
            f"{_iv['iv_annualised']:.1f}%",
            f"annualised · expiry {_iv.get('expiry','—')}",
            ACCENT_AMB,
            info=(
                "ATM Implied Volatility (IV) is the market's consensus forecast of future price "
                "volatility, derived from at-the-money options pricing. It is forward-looking — "
                "unlike historical (realised) volatility which looks backward. The IV cone on the "
                "chart shows the 1-standard-deviation (68%) and 2-sigma (95%) price ranges implied "
                "by the options market for this forecast horizon. When IV is high relative to "
                "recent realised vol, the market is pricing in elevated uncertainty."
            ),
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

    # IV cone (goes underneath everything)
    _iv_cone = iv_result or {}
    if _iv_cone.get("iv_cone_lo95"):
        _ivx = future_strs
        fc_traces += [
            _trace("Market IV (95%)", _ivx, _iv_cone["iv_cone_hi95"],
                   "#9ca3af", width=0, fill=None, show_legend=True),
            _trace("", _ivx, _iv_cone["iv_cone_lo95"],
                   "#9ca3af", width=0, fill="tonexty",
                   fill_color="#9ca3af11", show_legend=False),
            _trace("Market IV (80%)", _ivx, _iv_cone["iv_cone_hi80"],
                   "#9ca3af", width=0, fill=None, show_legend=False),
            _trace("", _ivx, _iv_cone["iv_cone_lo80"],
                   "#9ca3af", width=0, fill="tonexty",
                   fill_color="#9ca3af22", show_legend=False),
        ]

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

    if xgb_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + xgb_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "XGB 95% CI", "x": fx, "y": [conn_val[0]] + xgb_fc_hi95,
            "line": {"width": 0, "color": ACCENT_PRP},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "XGB 95% CI", "x": fx, "y": [conn_val[0]] + xgb_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_PRP + "22",
            "line": {"width": 0, "color": ACCENT_PRP},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "XGB 80% CI", "x": fx, "y": [conn_val[0]] + xgb_fc_hi80,
            "line": {"width": 0, "color": ACCENT_PRP},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "XGB 80% CI", "x": fx, "y": [conn_val[0]] + xgb_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_PRP + "44",
            "line": {"width": 0, "color": ACCENT_PRP},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "XGBoost", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_PRP, "width": 2, "dash": "dash"},
            "showlegend": True,
        })

    if prophet_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + prophet_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Prophet 95% CI", "x": fx, "y": [conn_val[0]] + prophet_fc_hi95,
            "line": {"width": 0, "color": ACCENT_ORG},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Prophet 95% CI", "x": fx, "y": [conn_val[0]] + prophet_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_ORG + "22",
            "line": {"width": 0, "color": ACCENT_ORG},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Prophet 80% CI", "x": fx, "y": [conn_val[0]] + prophet_fc_hi80,
            "line": {"width": 0, "color": ACCENT_ORG},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Prophet 80% CI", "x": fx, "y": [conn_val[0]] + prophet_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_ORG + "44",
            "line": {"width": 0, "color": ACCENT_ORG},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Prophet", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_ORG, "width": 2, "dash": "dot"},
            "showlegend": True,
        })

    if lstm_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + lstm_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "LSTM 95% CI", "x": fx, "y": [conn_val[0]] + lstm_fc_hi95,
            "line": {"width": 0, "color": ACCENT_FCS},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "LSTM 95% CI", "x": fx, "y": [conn_val[0]] + lstm_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_FCS + "22",
            "line": {"width": 0, "color": ACCENT_FCS},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "LSTM 80% CI", "x": fx, "y": [conn_val[0]] + lstm_fc_hi80,
            "line": {"width": 0, "color": ACCENT_FCS},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "LSTM 80% CI", "x": fx, "y": [conn_val[0]] + lstm_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_FCS + "44",
            "line": {"width": 0, "color": ACCENT_FCS},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "LSTM", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_FCS, "width": 2, "dash": "dashdot"},
            "showlegend": True,
        })

    if timesfm_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + timesfm_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "TimesFM 95% CI", "x": fx, "y": [conn_val[0]] + timesfm_fc_hi95,
            "line": {"width": 0, "color": ACCENT_CYN},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "TimesFM 95% CI", "x": fx, "y": [conn_val[0]] + timesfm_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_CYN + "22",
            "line": {"width": 0, "color": ACCENT_CYN},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "TimesFM 80% CI", "x": fx, "y": [conn_val[0]] + timesfm_fc_hi80,
            "line": {"width": 0, "color": ACCENT_CYN},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "TimesFM 80% CI", "x": fx, "y": [conn_val[0]] + timesfm_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_CYN + "44",
            "line": {"width": 0, "color": ACCENT_CYN},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "TimesFM", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_CYN, "width": 2, "dash": "dash"},
            "showlegend": True,
        })

    if chronos_fc_px:
        fx = conn_date + future_strs
        fy_mid = conn_val + chronos_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Chronos 95% CI", "x": fx, "y": [conn_val[0]] + chronos_fc_hi95,
            "line": {"width": 0, "color": ACCENT_LIM},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Chronos 95% CI", "x": fx, "y": [conn_val[0]] + chronos_fc_lo95,
            "fill": "tonexty", "fillcolor": ACCENT_LIM + "22",
            "line": {"width": 0, "color": ACCENT_LIM},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Chronos 80% CI", "x": fx, "y": [conn_val[0]] + chronos_fc_hi80,
            "line": {"width": 0, "color": ACCENT_LIM},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Chronos 80% CI", "x": fx, "y": [conn_val[0]] + chronos_fc_lo80,
            "fill": "tonexty", "fillcolor": ACCENT_LIM + "44",
            "line": {"width": 0, "color": ACCENT_LIM},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Chronos", "x": fx, "y": fy_mid,
            "line": {"color": ACCENT_LIM, "width": 2, "dash": "dot"},
            "showlegend": True,
        })

    if mc_fc_px:
        _mcx = future_strs[:len(mc_fc_px)]
        fc_traces += [
            _trace("MC (95% hi)",  _mcx, mc_fc_hi95[:len(_mcx)], ACCENT_GBM, width=0),
            _trace("MC (95% lo)", _mcx, mc_fc_lo95[:len(_mcx)], ACCENT_GBM, width=0,
                   fill="tonexty", fill_color=ACCENT_GBM+"15", show_legend=False),
            _trace("MC (80% hi)",  _mcx, mc_fc_hi80[:len(_mcx)], ACCENT_GBM, width=0, show_legend=False),
            _trace("MC (80% lo)", _mcx, mc_fc_lo80[:len(_mcx)], ACCENT_GBM, width=0,
                   fill="tonexty", fill_color=ACCENT_GBM+"25", show_legend=False),
            _trace(f"Monte Carlo (GBM)", _mcx, mc_fc_px[:len(_mcx)],
                   ACCENT_GBM, dash="dash", width=2),
        ]

    if tft_fc_px:
        _tftx = future_strs[:len(tft_fc_px)]
        fc_traces += [
            _trace("TFT (95% hi)",  _tftx, tft_fc_hi95[:len(_tftx)], ACCENT_TFT, width=0),
            _trace("TFT (95% lo)", _tftx, tft_fc_lo95[:len(_tftx)], ACCENT_TFT, width=0,
                   fill="tonexty", fill_color=ACCENT_TFT+"15", show_legend=False),
            _trace("TFT (80% hi)",  _tftx, tft_fc_hi80[:len(_tftx)], ACCENT_TFT, width=0, show_legend=False),
            _trace("TFT (80% lo)", _tftx, tft_fc_lo80[:len(_tftx)], ACCENT_TFT, width=0,
                   fill="tonexty", fill_color=ACCENT_TFT+"25", show_legend=False),
            _trace("TFT", _tftx, tft_fc_px[:len(_tftx)],
                   ACCENT_TFT, dash="dash", width=2),
        ]

    if nhits_fc_px:
        _nhtx = future_strs[:len(nhits_fc_px)]
        fc_traces += [
            _trace("N-HiTS (95% hi)",  _nhtx, nhits_fc_hi95[:len(_nhtx)], ACCENT_NHT, width=0),
            _trace("N-HiTS (95% lo)", _nhtx, nhits_fc_lo95[:len(_nhtx)], ACCENT_NHT, width=0,
                   fill="tonexty", fill_color=ACCENT_NHT+"15", show_legend=False),
            _trace("N-HiTS (80% hi)",  _nhtx, nhits_fc_hi80[:len(_nhtx)], ACCENT_NHT, width=0, show_legend=False),
            _trace("N-HiTS (80% lo)", _nhtx, nhits_fc_lo80[:len(_nhtx)], ACCENT_NHT, width=0,
                   fill="tonexty", fill_color=ACCENT_NHT+"25", show_legend=False),
            _trace("N-HiTS", _nhtx, nhits_fc_px[:len(_nhtx)],
                   ACCENT_NHT, dash="dash", width=2),
        ]

    if nbeats_fc_px:
        _nbtx = future_strs[:len(nbeats_fc_px)]
        fc_traces += [
            _trace("N-BEATS (95% hi)",  _nbtx, nbeats_fc_hi95[:len(_nbtx)], ACCENT_NBT, width=0),
            _trace("N-BEATS (95% lo)", _nbtx, nbeats_fc_lo95[:len(_nbtx)], ACCENT_NBT, width=0,
                   fill="tonexty", fill_color=ACCENT_NBT+"15", show_legend=False),
            _trace("N-BEATS (80% hi)",  _nbtx, nbeats_fc_hi80[:len(_nbtx)], ACCENT_NBT, width=0, show_legend=False),
            _trace("N-BEATS (80% lo)", _nbtx, nbeats_fc_lo80[:len(_nbtx)], ACCENT_NBT, width=0,
                   fill="tonexty", fill_color=ACCENT_NBT+"25", show_legend=False),
            _trace("N-BEATS", _nbtx, nbeats_fc_px[:len(_nbtx)],
                   ACCENT_NBT, dash="dashdot", width=2),
        ]

    if ens_fc_px:
        _ENS_CLR = "#ffffff"
        fx = conn_date + future_strs
        fy_mid = conn_val + ens_fc_px
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Ensemble 95% CI", "x": fx, "y": [conn_val[0]] + ens_fc_hi95,
            "line": {"width": 0, "color": _ENS_CLR},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Ensemble 95% CI", "x": fx, "y": [conn_val[0]] + ens_fc_lo95,
            "fill": "tonexty", "fillcolor": _ENS_CLR + "18",
            "line": {"width": 0, "color": _ENS_CLR},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Ensemble 80% CI", "x": fx, "y": [conn_val[0]] + ens_fc_hi80,
            "line": {"width": 0, "color": _ENS_CLR},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Ensemble 80% CI", "x": fx, "y": [conn_val[0]] + ens_fc_lo80,
            "fill": "tonexty", "fillcolor": _ENS_CLR + "30",
            "line": {"width": 0, "color": _ENS_CLR},
            "showlegend": False,
        })
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Ensemble", "x": fx, "y": fy_mid,
            "line": {"color": _ENS_CLR, "width": 2.5},
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

    # ── Fundamental target lines (horizontal, spanning history + forecast) ──────
    _full_x = [hist_dates[0], future_strs[-1]] if future_strs else hist_dates[:1]*2
    if _fres.get("models"):
        # Individual model lines — subtle, toggle-able from legend
        _model_color = BORDER   # very muted so chart isn't cluttered
        for _m in _fres["models"]:
            _fv = _m.get("fair_value")
            if not (_fv and _fv > 0):
                continue
            fc_traces.append({
                "type": "scatter", "mode": "lines",
                "name": _m["method"],
                "x": _full_x, "y": [_fv, _fv],
                "line": {"color": "#4a5568", "width": 1, "dash": "longdash"},
                "showlegend": True,
                "legendgroup": "fundamental",
                "legendgrouptitle": {"text": "Fundamental Targets"},
                "hovertemplate": (
                    f"<b>{_m['method']}</b>: {currency} {_fv:,.2f}"
                    + (f" (MoS: {currency} {_m['mos_value']:,.2f})" if _m.get('mos_value') else "")
                    + "<extra></extra>"
                ),
            })

        # 25–75 pct fair value range as a shaded band
        _flow = _fres.get("low")
        _fhigh = _fres.get("high")
        if _flow and _fhigh and _fhigh > _flow:
            fc_traces.append({
                "type": "scatter", "mode": "lines",
                "name": "FV range (25–75 pct)",
                "x": _full_x, "y": [_fhigh, _fhigh],
                "line": {"width": 0, "color": ACCENT_TEA},
                "showlegend": False,
                "legendgroup": "fundamental",
                "hoverinfo": "skip",
            })
            fc_traces.append({
                "type": "scatter", "mode": "lines",
                "name": "FV Range (IQR)",
                "x": _full_x, "y": [_flow, _flow],
                "fill": "tonexty", "fillcolor": ACCENT_TEA + "18",
                "line": {"width": 0, "color": ACCENT_TEA},
                "showlegend": True,
                "legendgroup": "fundamental",
                "hoverinfo": "skip",
            })

        # Composite target — prominent solid line
        if _fcomp:
            fc_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"Fundamental Composite ({currency} {_fcomp:,.2f})",
                "x": _full_x, "y": [_fcomp, _fcomp],
                "line": {"color": ACCENT_TEA, "width": 2.5, "dash": "dash"},
                "showlegend": True,
                "legendgroup": "fundamental",
                "hovertemplate": (
                    f"<b>Fundamental Composite</b>: {currency} {_fcomp:,.2f}<extra></extra>"
                ),
            })

    # Analyst consensus target
    if _analyst_tgt:
        fc_traces.append({
            "type": "scatter", "mode": "lines",
            "name": f"Analyst Target ({currency} {_analyst_tgt:,.2f})",
            "x": _full_x, "y": [_analyst_tgt, _analyst_tgt],
            "line": {"color": "#f472b6", "width": 1.5, "dash": "dot"},
            "showlegend": True,
            "hovertemplate": (
                f"<b>Analyst Consensus</b>: {currency} {_analyst_tgt:,.2f}<extra></extra>"
            ),
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

    # ── Fundamental Value Targets HTML section ────────────────────────────────
    _fundamental_html = ""
    if _fres.get("models"):
        _trows = ""
        for _m in _fres["models"]:
            _fv  = _m.get("fair_value", 0)
            _mos = _m.get("mos_value", 0)
            _rel = _m.get("reliable", True)
            if not _fv:
                continue
            _upside = (_fv / last_price - 1) * 100 if last_price > 0 else 0
            _uclr   = ACCENT_GRN if _upside > 0 else ACCENT_RED
            _rflag  = "" if _rel else f' <span style="color:{ACCENT_AMB};font-size:10px">⚠ low reliability</span>'
            _note   = f'<span style="color:{TEXT_SUB};font-size:10px">{_m["note"][:60]}</span>' if _m.get("note") else ""
            _trows += (
                f'<tr>'
                f'<td style="font-weight:500">{_m["method"]}{_rflag}</td>'
                f'<td style="font-weight:600">{currency} {_fv:,.2f}</td>'
                f'<td style="color:{_uclr};font-weight:600">{"+" if _upside >= 0 else ""}{_upside:.1f}%</td>'
                f'<td style="color:{TEXT_SUB}">{currency} {_mos:,.2f}</td>'
                f'<td>{_note}</td>'
                f'</tr>'
            )

        # Composite row
        if _fcomp:
            _cup = (_fcomp / last_price - 1) * 100 if last_price > 0 else 0
            _cuclr = ACCENT_GRN if _cup > 0 else ACCENT_RED
            _n_models = len(_fres["models"])
            _trows += (
                f'<tr style="border-top:2px solid {ACCENT_TEA};font-size:13px">'
                f'<td style="font-weight:700;color:{ACCENT_TEA}">Composite (median, {_n_models} models)</td>'
                f'<td style="font-weight:700;color:{ACCENT_TEA}">{currency} {_fcomp:,.2f}</td>'
                f'<td style="font-weight:700;color:{_cuclr}">{"+" if _cup >= 0 else ""}{_cup:.1f}%</td>'
                f'<td style="color:{TEXT_SUB}">{currency} {_fcomp * 0.80:,.2f}</td>'
                f'<td style="color:{TEXT_SUB}">median of {_n_models} valid models</td>'
                f'</tr>'
            )

        # Analyst consensus row
        if _analyst_tgt:
            _aup = (_analyst_tgt / last_price - 1) * 100 if last_price > 0 else 0
            _auclr = ACCENT_GRN if _aup > 0 else ACCENT_RED
            _trows += (
                f'<tr>'
                f'<td style="font-weight:500;color:#f472b6">Analyst Consensus</td>'
                f'<td style="font-weight:600">{currency} {_analyst_tgt:,.2f}</td>'
                f'<td style="color:{_auclr};font-weight:600">{"+" if _aup >= 0 else ""}{_aup:.1f}%</td>'
                f'<td style="color:{TEXT_SUB}">—</td>'
                f'<td style="color:{TEXT_SUB};font-size:10px">Wall Street consensus mean</td>'
                f'</tr>'
            )

        _sector_note = _fres["models"][0].get("note", "") if _fres["models"] else ""
        _flo = _fres.get("low")
        _fhi = _fres.get("high")
        _range_str = (f"IQR range: {currency} {_flo:,.2f} – {currency} {_fhi:,.2f}"
                      if _flo and _fhi else "")

        _fundamental_html = f"""
<!-- Fundamental Value Targets -->
<div class="card" style="margin-top:16px">
  <h3 style="margin-bottom:4px">Fundamental Value Targets
    <span class="note" style="font-size:11px;font-weight:normal;margin-left:8px">
      from the Valuation Suite — using sector benchmarks where peer data is unavailable
    </span>
  </h3>
  <p class="note" style="margin-bottom:12px">
    These are intrinsic value estimates from DCF, relative, and income-based models.
    Horizontal dashed lines on the forecast chart show the same targets.
    {_range_str}
  </p>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Fair Value</th>
        <th>vs Current Price</th>
        <th>Margin of Safety (80%)</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>{_trows}</tbody>
  </table>
</div>"""

    # ── Regime note ───────────────────────────────────────────────────────────
    _regime_note = ""
    if regime_result and regime_result.get("note"):
        _regime_clr2 = {"Bull": ACCENT_GRN, "Bear": ACCENT_RED}.get(
            regime_result.get("regime",""), TEXT_SUB)
        _regime_note = (f'<p class="note" style="margin-top:6px;color:{_regime_clr2}">'
                        f'{regime_result["note"]}</p>')

    # ── Analyst estimate revisions ─────────────────────────────────────────────
    _revisions_html = ""
    _er = estimate_revisions or {}
    if _er.get("eps") or _er.get("revenue"):
        def _rev_row(rev):
            if not rev:
                return ""
            d = rev.get("direction","stable")
            arrow = {"improving": "▲", "deteriorating": "▼"}.get(d, "→")
            clr   = {"improving": ACCENT_GRN, "deteriorating": ACCENT_RED}.get(d, TEXT_SUB)
            mag   = f'{rev["magnitude"]:+.1f}%' if rev.get("magnitude") is not None else "—"
            cur   = f'{rev.get("current", "—")}'
            ago90 = f'{rev.get("ago90","—")}'
            return (f'<tr><td style="font-weight:500">{rev.get("label","")}</td>'
                    f'<td>{cur}</td><td>{ago90}</td>'
                    f'<td style="color:{clr};font-weight:600">{arrow} {mag}</td></tr>')
        _revisions_html = f'''
<div class="card" style="margin-top:16px">
  <h3 style="margin:0 0 12px;font-size:14px;color:{TEXT_MAIN}">
    Analyst Estimate Revisions
    <span style="font-size:11px;color:{TEXT_SUB};font-weight:400;margin-left:8px">
      vs 90 days ago
    </span>
  </h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead>
      <tr style="color:{TEXT_SUB};font-size:11px;border-bottom:1px solid {BORDER}">
        <th style="text-align:left;padding:4px 8px">Metric</th>
        <th style="text-align:left;padding:4px 8px">Current Estimate</th>
        <th style="text-align:left;padding:4px 8px">90d Ago</th>
        <th style="text-align:left;padding:4px 8px">Revision</th>
      </tr>
    </thead>
    <tbody>
      {_rev_row(_er.get("eps"))}
      {_rev_row(_er.get("revenue"))}
    </tbody>
  </table>
  <p class="note" style="margin-top:8px">{_er.get("summary","")}<br>
  Source: Yahoo Finance analyst consensus estimates.</p>
</div>'''

    # ── Multi-horizon milestone panel ─────────────────────────────────────────
    _mh_forecasts = {}
    for _name, _fc in [
        ("ARIMA",       arima_fc_px),    ("ETS",    ets_fc_px),
        ("XGB",         xgb_fc_px),      ("Prophet",prophet_fc_px),
        ("LSTM",        lstm_fc_px),     ("MC",     mc_fc_px),
        ("TFT",         tft_fc_px),      ("NHiTS",  nhits_fc_px),
        ("NBEATS",      nbeats_fc_px),   ("Ensemble",ens_fc_px),
    ]:
        if _fc:
            _mh_forecasts[_name] = _fc
    _mh_rows = compute_multihorizon(_mh_forecasts, horizon) if _mh_forecasts else []

    _multihorizon_html = ""
    if len(_mh_rows) >= 2:
        _mh_days    = [r["day"]    for r in _mh_rows]
        _mh_medians = [r["median"] for r in _mh_rows]
        _mh_q25     = [r["q25"]    for r in _mh_rows]
        _mh_q75     = [r["q75"]    for r in _mh_rows]
        _mh_mins    = [r["min"]    for r in _mh_rows]
        _mh_maxs    = [r["max"]    for r in _mh_rows]
        _multihorizon_html = f'''
<div class="card" style="margin-top:16px">
  <h3 style="margin:0 0 4px;font-size:14px;color:{TEXT_MAIN}">
    Multi-Horizon Forecast Distribution
    <span style="font-size:11px;color:{TEXT_SUB};font-weight:400;margin-left:8px">
      ensemble of all active models &middot; milestone trading days
    </span>
  </h3>
  <div id="chart-multihorizon" class="chart-wrap-sm"></div>
</div>'''

    # ── STL decomposition panel ────────────────────────────────────────────────
    _stl_html = ""
    _stl = stl_result or {}
    if _stl.get("dates"):
        _st = _stl["strength_trend"]
        _ss = _stl["strength_seasonal"]
        _stl_html = f'''
<div class="card" style="margin-top:16px">
  <h3 style="margin:0 0 4px;font-size:14px;color:{TEXT_MAIN}">
    STL Decomposition
    <span style="font-size:11px;color:{TEXT_SUB};font-weight:400;margin-left:8px">
      Trend strength {_st:.0%} &middot; Seasonal strength {_ss:.0%} &middot; log-price basis
    </span>
  </h3>
  <div id="chart-stl-trend" style="height:120px"></div>
  <div id="chart-stl-seasonal" style="height:100px"></div>
  <div id="chart-stl-residual" style="height:100px"></div>
</div>'''

    # ── Scenario table ─────────────────────────────────────────────────────────
    _scenario_html = ""
    if len(_endpoint_vals) >= 2:
        _sv = sorted(_endpoint_vals)
        _n  = len(_sv)
        _bear_price  = round(_sv[0], 2)
        _base_price  = round(_sv[_n // 2] if _n % 2 == 1
                             else (_sv[_n//2-1] + _sv[_n//2]) / 2, 2)
        _bull_price  = round(_sv[-1], 2)

        def _nearest_model(target):
            closest = min(
                [(name, val) for name, val in zip(
                    ["ARIMA","ETS","XGB","Prophet","LSTM","TimesFM","Chronos","MC","TFT","N-HiTS","N-BEATS"],
                    [arima_end,ets_end,xgb_end,prophet_end,lstm_end,
                     timesfm_end,chronos_end,mc_end,tft_end,nhits_end,nbeats_end]
                ) if val is not None],
                key=lambda x: abs(x[1] - target),
                default=("—", 0)
            )
            return closest[0]

        def _scen_row(label, price, icon, row_color):
            chg   = (price / last_price - 1) * 100
            chg_s = f'{"+" if chg >= 0 else ""}{chg:.1f}%'
            clr   = ACCENT_GRN if chg >= 0 else ACCENT_RED
            model = _nearest_model(price)
            return (f'<tr style="border-bottom:1px solid {BORDER}">'
                    f'<td style="padding:8px 12px;font-weight:600;color:{row_color}">{icon} {label}</td>'
                    f'<td style="padding:8px 12px;font-weight:700">{currency} {price:,.2f}</td>'
                    f'<td style="padding:8px 12px;font-weight:600;color:{clr}">{chg_s}</td>'
                    f'<td style="padding:8px 12px;color:{TEXT_SUB};font-size:12px">{model}</td>'
                    f'</tr>')

        _scenario_html = f'''
<div class="card" style="margin-top:16px">
  <h3 style="margin:0 0 12px;font-size:14px;color:{TEXT_MAIN}">
    Scenario Analysis
    <span style="font-size:11px;color:{TEXT_SUB};font-weight:400;margin-left:8px">
      t+{horizon}d &middot; bull/base/bear from model distribution
    </span>
  </h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead>
      <tr style="color:{TEXT_SUB};font-size:11px;border-bottom:1px solid {BORDER}">
        <th style="text-align:left;padding:6px 12px">Scenario</th>
        <th style="text-align:left;padding:6px 12px">Price Target</th>
        <th style="text-align:left;padding:6px 12px">vs Today</th>
        <th style="text-align:left;padding:6px 12px">Driven by</th>
      </tr>
    </thead>
    <tbody>
      {_scen_row("Bull", _bull_price, "&#9650;", ACCENT_GRN)}
      {_scen_row("Base", _base_price, "&#8594;", ACCENT_TEA)}
      {_scen_row("Bear", _bear_price, "&#9660;", ACCENT_RED)}
    </tbody>
  </table>
  <p class="note" style="margin-top:8px">
    Bull = most optimistic model endpoint &middot; Base = median of all models &middot; Bear = most pessimistic.
    These are price-level forecasts from statistical and ML models, not fundamental valuations.
  </p>
</div>'''

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
                 border-radius:8px; padding:14px 18px; min-width:140px;
                 position:relative; }}
  .stat-val   {{ font-size:20px; font-weight:700; line-height:1.2; }}
  .stat-sub   {{ font-size:11px; color:{TEXT_SUB}; margin-top:2px; }}
  .stat-note  {{ font-size:12px; font-weight:600; margin-top:4px; }}
  .help-btn   {{ position:absolute; top:7px; right:8px; background:none;
                 border:none; color:{TEXT_SUB}; font-size:13px;
                 cursor:pointer; padding:0; line-height:1; }}
  .help-btn:hover {{ color:{TEXT_MAIN}; }}
  #help-pop   {{ display:none; position:fixed; z-index:9999;
                 background:{CARD_BG}; border:1px solid {ACCENT_TEA};
                 border-radius:8px; padding:18px 18px 14px;
                 max-width:380px; width:min(380px,90vw);
                 box-shadow:0 8px 40px rgba(0,0,0,.65);
                 font-size:12.5px; line-height:1.6; color:{TEXT_MAIN}; }}
  #help-pop-title {{ font-weight:700; font-size:14px; color:{ACCENT_TEA};
                     margin-bottom:10px; padding-bottom:8px;
                     border-bottom:1px solid {BORDER}; }}
  #help-pop-close {{ position:absolute; top:10px; right:12px; background:none;
                     border:none; color:{TEXT_SUB}; font-size:20px;
                     line-height:1; cursor:pointer; padding:0; }}
  #help-pop-close:hover {{ color:{TEXT_MAIN}; }}
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
    <span class="badge">{_badge_models}</span>
    &nbsp;
    {_regime_badge}
    &nbsp;
    {_disagree_badge}
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

{_multihorizon_html}

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

{_stl_html}

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
  {_regime_note}
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

{_fundamental_html}
{_revisions_html}
{_scenario_html}

<!-- Disclaimer -->
<div class="disclaimer">
  <b>⚠ Important:</b> These forecasts are statistical model outputs, not investment advice.
  Financial markets are widely believed to follow a random walk under the
  Efficient Market Hypothesis — past prices contain limited information about future prices.
  ARIMA and ETS models may capture short-term autocorrelation or trend momentum but
  typically do <b>not</b> outperform naive baselines on a risk-adjusted basis over long horizons.
  Fundamental targets are intrinsic-value estimates, not price predictions — convergence
  may take months or years and may never occur.
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

// Multi-horizon chart
(function() {{
  if (document.getElementById('chart-multihorizon')) {{
    var mhData = __MHDATA__;
    if (!mhData || !mhData.days || mhData.days.length < 2) return;
    var days = mhData.days, medians = mhData.medians, q25 = mhData.q25,
        q75 = mhData.q75, mins = mhData.mins, maxs = mhData.maxs, lp = mhData.last_price;
    var traces = [
      {{ type:'scatter', x:days, y:maxs, mode:'lines', line:{{width:0}}, showlegend:false, name:'Range hi' }},
      {{ type:'scatter', x:days, y:mins, mode:'lines', line:{{width:0}}, fill:'tonexty',
         fillcolor:'{ACCENT_TEA}18', showlegend:false, name:'Full range' }},
      {{ type:'scatter', x:days, y:q75, mode:'lines', line:{{width:0}}, showlegend:false, name:'IQR hi' }},
      {{ type:'scatter', x:days, y:q25, mode:'lines', line:{{width:0}}, fill:'tonexty',
         fillcolor:'{ACCENT_TEA}44', name:'IQR (25–75%)', showlegend:true }},
      {{ type:'scatter', x:days, y:medians, mode:'lines+markers',
         line:{{color:'{ACCENT_TEA}', width:2}},
         marker:{{color:'{ACCENT_TEA}', size:7}}, name:'Median forecast' }},
      {{ type:'scatter', x:days, y:days.map(function(){{return lp;}}), mode:'lines',
         line:{{color:'{TEXT_SUB}', width:1, dash:'dot'}}, name:'Current price', showlegend:true }},
    ];
    var layout = {{
      paper_bgcolor:'{DARK_BG}', plot_bgcolor:'{CARD_BG}',
      font:{{color:'{TEXT_MAIN}', size:11}},
      margin:{{l:60,r:20,t:10,b:40}},
      xaxis:{{title:'Trading days ahead', gridcolor:'{BORDER}', color:'{TEXT_SUB}'}},
      yaxis:{{title:'Price ({currency})', gridcolor:'{BORDER}', color:'{TEXT_SUB}',
              tickprefix:'{currency} ', tickformat:',.0f'}},
      legend:{{bgcolor:'{CARD_BG}', bordercolor:'{BORDER}', borderwidth:1}},
      shapes:[{{type:'line', x0:days[0], x1:days[days.length-1],
                y0:lp, y1:lp, line:{{color:'{TEXT_SUB}', width:1, dash:'dot'}}}}]
    }};
    Plotly.newPlot('chart-multihorizon', traces, layout, plotlyConfig);
  }}
}})();

// STL decomposition charts
(function() {{
  var stlData = __STLDATA__;
  if (!stlData || !stlData.dates) return;
  var dates = stlData.dates, trend = stlData.trend,
      seasonal = stlData.seasonal, residual = stlData.residual;
  var baseLayout = {{
    paper_bgcolor:'{DARK_BG}', plot_bgcolor:'{CARD_BG}',
    font:{{color:'{TEXT_MAIN}', size:10}},
    margin:{{l:55,r:10,t:6,b:30}},
    xaxis:{{gridcolor:'{BORDER}', color:'{TEXT_SUB}', showticklabels:true}},
    yaxis:{{gridcolor:'{BORDER}', color:'{TEXT_SUB}'}},
    showlegend:false, hovermode:'x unified',
  }};
  if (document.getElementById('chart-stl-trend')) {{
    Plotly.newPlot('chart-stl-trend',
      [{{type:'scatter', mode:'lines', x:dates, y:trend, name:'Trend (price)',
         line:{{color:'{ACCENT_TEA}', width:1.5}}}}],
      Object.assign({{}}, baseLayout, {{yaxis:{{title:'Trend (price)', gridcolor:'{BORDER}', color:'{TEXT_SUB}', tickprefix:'{currency} '}}}}),
      plotlyConfig);
  }}
  if (document.getElementById('chart-stl-seasonal')) {{
    Plotly.newPlot('chart-stl-seasonal',
      [{{type:'scatter', mode:'lines', x:dates, y:seasonal, name:'Seasonal',
         line:{{color:'{ACCENT_AMB}', width:1}}}}],
      Object.assign({{}}, baseLayout, {{yaxis:{{title:'Seasonal', gridcolor:'{BORDER}', color:'{TEXT_SUB}'}}}}),
      plotlyConfig);
  }}
  if (document.getElementById('chart-stl-residual')) {{
    Plotly.newPlot('chart-stl-residual',
      [{{type:'scatter', mode:'lines', x:dates, y:residual, name:'Residual',
         line:{{color:'{ACCENT_GBM}', width:1}}}}],
      Object.assign({{}}, baseLayout, {{yaxis:{{title:'Residual', gridcolor:'{BORDER}', color:'{TEXT_SUB}'}}}}),
      plotlyConfig);
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

<!-- Shared help popover -->
<div id="help-pop" role="tooltip">
  <button id="help-pop-close" onclick="closeHelp()" aria-label="Close">&times;</button>
  <div id="help-pop-title"></div>
  <div id="help-pop-body"></div>
</div>
<script>
var _HELP=__HELPDATA__;
function showHelp(key,evt){{
  evt.stopPropagation();
  var d=_HELP[key]; if(!d) return;
  var pop=document.getElementById('help-pop');
  document.getElementById('help-pop-title').textContent=d.title;
  document.getElementById('help-pop-body').innerHTML=d.body;
  pop.style.visibility='hidden'; pop.style.display='block';
  var vw=window.innerWidth,vh=window.innerHeight,pw=pop.offsetWidth,ph=pop.offsetHeight;
  var x=evt.clientX+14,y=evt.clientY+14;
  if(x+pw>vw-8) x=evt.clientX-pw-14;
  if(x<8) x=8;
  if(y+ph>vh-8) y=evt.clientY-ph-14;
  if(y<8) y=8;
  pop.style.left=x+'px'; pop.style.top=y+'px'; pop.style.visibility='';
}}
function closeHelp(){{ document.getElementById('help-pop').style.display='none'; }}
document.addEventListener('click',function(e){{
  var pop=document.getElementById('help-pop');
  if(pop&&pop.style.display!=='none'&&!pop.contains(e.target)) closeHelp();
}});
document.addEventListener('keydown',function(e){{ if(e.key==='Escape') closeHelp(); }});
</script>
</body>
</html>
"""
    import json as _json
    html = html.replace("__HELPDATA__", _json.dumps(_help_data))
    # Multi-horizon data
    _mh_payload = {}
    if _mh_rows:
        _mh_payload = {
            "days":       [r["day"]    for r in _mh_rows],
            "medians":    [r["median"] for r in _mh_rows],
            "q25":        [r["q25"]    for r in _mh_rows],
            "q75":        [r["q75"]    for r in _mh_rows],
            "mins":       [r["min"]    for r in _mh_rows],
            "maxs":       [r["max"]    for r in _mh_rows],
            "last_price": last_price,
        }
    html = html.replace("__MHDATA__", _json.dumps(_mh_payload))
    # STL data
    _stl_payload = {}
    if _stl.get("dates"):
        _stl_payload = {
            "dates":    _stl["dates"],
            "trend":    _stl["trend"],
            "seasonal": _stl["seasonal"],
            "residual": _stl["residual"],
        }
    html = html.replace("__STLDATA__", _json.dumps(_stl_payload))
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Price / return forecasting tool")
    ap.add_argument("--ticker",  default="AAPL")
    ap.add_argument("--horizon", type=int, default=30,
                    help="Trading days to forecast (default 30)")
    ap.add_argument("--model",   default="ensemble",
                    choices=["arima", "ets", "both", "xgb", "prophet", "lstm",
                             "timesfm", "chronos", "montecarlo", "tft",
                             "nhits", "nbeats",
                             "ensemble", "all"])
    ap.add_argument("--period",  default="5y",
                    choices=["2y", "5y", "10y"])
    args = ap.parse_args()

    ticker  = args.ticker.upper().strip()
    horizon = max(5, min(args.horizon, 252))
    do_arima    = args.model in ("arima", "both", "all", "ensemble")
    do_ets      = args.model in ("ets",   "both", "all", "ensemble")
    do_xgb      = args.model in ("xgb", "ensemble", "all") and XGB_OK
    do_ensemble = args.model in ("ensemble", "all")
    do_prophet  = args.model in ("prophet", "ensemble", "all") and PROPHET_OK
    do_lstm     = args.model in ("lstm", "ensemble", "all") and TORCH_OK
    do_timesfm  = args.model in ("timesfm", "all") and TIMESFM_OK
    do_chronos  = args.model in ("chronos", "all") and CHRONOS_OK
    do_montecarlo = args.model in ("montecarlo", "ensemble", "all")
    do_tft        = args.model in ("tft", "all") and TFT_OK
    do_nhits_nbeats = args.model in ("nhits", "nbeats", "ensemble", "all") and NEURALFORECAST_OK

    if do_arima and not PMDARIMA_OK:
        print("[warn] pmdarima not installed — falling back to ETS only.")
        do_arima = False
        do_ets   = True

    _active = [m for m, ok in [
        ("ARIMA", do_arima), ("ETS", do_ets), ("XGB", do_xgb),
        ("Prophet", do_prophet), ("LSTM", do_lstm),
        ("TimesFM", do_timesfm), ("Chronos", do_chronos),
        ("Monte Carlo", do_montecarlo), ("TFT", do_tft),
        ("N-HiTS", do_nhits_nbeats), ("N-BEATS", do_nhits_nbeats),
        ("Ensemble", do_ensemble),
    ] if ok]
    print(f"\n{'='*60}", flush=True)
    print(f"  Price Forecast · {ticker} · horizon={horizon}d · "
          f"model={'+'.join(_active) or 'None'}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Data
    prices = fetch_prices(ticker, args.period)
    info   = fetch_info(ticker)

    regime_result = detect_regime(prices)
    print(f"  Regime: {regime_result['regime']}", flush=True)

    # Fundamental valuation — runs in parallel with statistical fits
    fund_d       = fetch_fundamentals(ticker)
    fund_results = run_fundamental_valuations(fund_d) if fund_d else {}

    estimate_revisions = fetch_estimate_revisions(ticker)
    iv_result = fetch_iv_data(ticker, prices, horizon)
    if iv_result.get("iv_annualised"):
        print(f"  ATM IV: {iv_result['iv_annualised']:.1f}% (expiry {iv_result['expiry']})", flush=True)

    macro_df = None
    if do_xgb or do_lstm or do_tft:
        try:
            sector_name = info.get("sector", "")
            macro_df = fetch_macro_features(prices, sector_name, args.period)
            print(f"  Macro features: {len([c for c in macro_df.columns if macro_df[c].notna().any()])} cols fetched", flush=True)
        except Exception as _me:
            print(f"  [warn] Macro fetch failed: {_me} — using price-only features", flush=True)

    n_steps = ((2 if do_arima else 0) + (1 if do_ets else 0) + 2
               + (1 if do_xgb else 0) + (1 if do_ensemble else 0))
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

    # 5b. XGBoost forecast
    xgb_result = {}
    if do_xgb:
        _step("XGBoost walk-forward forecast")
        xgb_result = fit_xgb_forecast(prices, horizon, macro_df=macro_df)
        _mape = xgb_result.get("backtest_mape")
        print(f"  XGB MAPE: {_mape:.2f}%" if _mape else "  XGB MAPE: n/a (short history)",
              flush=True)

    # 5c. Prophet forecast
    prophet_result = {}
    if do_prophet:
        _step("Prophet (additive seasonality model)")
        prophet_result = fit_prophet_forecast(prices, horizon)
        _pm = prophet_result.get("backtest_mape")
        print(f"  Prophet MAPE: {_pm:.2f}%" if _pm else "  Prophet MAPE: n/a", flush=True)

    # 5d. LSTM forecast
    lstm_result = {}
    if do_lstm:
        _step("LSTM (2-layer, MC-Dropout CI)")
        lstm_result = fit_lstm_forecast(prices, horizon, macro_df=macro_df)
        _lm = lstm_result.get("backtest_mape")
        print(f"  LSTM MAPE: {_lm:.2f}%" if _lm else "  LSTM MAPE: n/a", flush=True)

    # 5e. TimesFM forecast (zero-shot)
    timesfm_result = {}
    if do_timesfm:
        _step("TimesFM-1.0-200M (zero-shot)")
        timesfm_result = fit_timesfm_forecast(prices, horizon)
        print(f"  TimesFM: {'OK' if timesfm_result else 'failed'}", flush=True)

    # 5f. Chronos forecast (zero-shot)
    chronos_result = {}
    if do_chronos:
        _step("Chronos-Bolt-Small (zero-shot)")
        chronos_result = fit_chronos_forecast(prices, horizon)
        print(f"  Chronos: {'OK' if chronos_result else 'failed'}", flush=True)

    # 5g. Monte Carlo GBM
    montecarlo_result = {}
    if do_montecarlo:
        n_mc_steps = n_steps
        print(f"\n[{n_mc_steps}/{n_steps}] Monte Carlo GBM (1000 simulations)", flush=True)
        montecarlo_result = fit_montecarlo_forecast(prices, horizon)

    # 5h. TFT
    tft_result = {}
    if do_tft:
        print(f"\n[TFT] Temporal Fusion Transformer", flush=True)
        tft_result = fit_tft_forecast(prices, horizon, macro_df=macro_df)

    # 5i. N-HiTS / N-BEATS
    nhits_result = {}
    nbeats_result = {}
    if do_nhits_nbeats:
        print(f"\n[N-HiTS/N-BEATS] Neural basis expansion models", flush=True)
        nhits_result, nbeats_result = fit_nhits_nbeats_forecast(prices, horizon)

    # STL decomposition (always, uses existing prices)
    stl_result = compute_stl(prices)

    # 5j. Ensemble blend hint (actual blending done in build_html after ARIMA/ETS arrays computed)
    if do_ensemble:
        _step("Ensemble blend (inverse-MAPE weighted)")

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
        xgb_result=xgb_result,
        do_ensemble=do_ensemble,
        prophet_result=prophet_result,
        lstm_result=lstm_result,
        timesfm_result=timesfm_result,
        chronos_result=chronos_result,
        fundamental_results=fund_results,
        montecarlo_result=montecarlo_result,
        iv_result=iv_result,
        tft_result=tft_result,
        regime_result=regime_result,
        estimate_revisions=estimate_revisions,
        nhits_result=nhits_result,
        nbeats_result=nbeats_result,
        stl_result=stl_result,
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
