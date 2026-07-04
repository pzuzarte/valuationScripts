"""
Portfolio Optimizer -- Efficient Frontier & Correlation
=======================================================
Mean-variance optimization over a candidate set of tickers.  Builds the
efficient frontier, finds the maximum-Sharpe (tangency) and minimum-variance
portfolios, and shows the correlation structure -- the raw material for
diversification decisions.

Method
------
  * Daily returns from ~3 years of price history (yfinance).
  * Annualised expected return  mu  = mean(daily) * 252
  * Annualised covariance       cov = cov(daily) * 252
  * Long-only, fully-invested (weights >= 0, sum = 1), via scipy SLSQP.
  * Frontier: minimise variance across a grid of target returns.
  * Reuses the suite's calc_portfolio_risk() for the recommended portfolio's
    VaR / max-drawdown / Calmar.

USAGE
-----
  python portfolioOptimizer.py NVDA,MSFT,AAPL,GOOGL,AMZN
  python portfolioOptimizer.py AAPL,KO,XOM,JNJ,NVDA --open
  python portfolioOptimizer.py --tickers SPY,TLT,GLD,QQQ --years 5

Caveat: expected returns from historical means are notoriously noisy; treat
max-Sharpe weights as a starting point, not gospel.  The correlation structure
and the frontier *shape* are far more robust than the point estimates.
"""

import sys, os, io, argparse, base64, datetime, webbrowser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from valuation_models import calc_portfolio_risk
except Exception:
    calc_portfolio_risk = None

BG = "#0f1117"; S1 = "#161a27"; S2 = "#1a1e2e"; BD = "#252a3a"
TEXT = "#e2e8f0"; SUB = "#94a3b8"
GRN = "#22c55e"; LINE1 = "#6366f1"; LINE2 = "#14b8a6"; LINE3 = "#f59e0b"; RED = "#ef4444"


# ── § Data ────────────────────────────────────────────────────────────────────
def fetch_prices(tickers, years=3):
    raw = yf.download(tickers, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return None
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all").ffill().dropna(axis=1, how="any")
    # keep only tickers with a full history
    good = [t for t in tickers if t in close.columns]
    return close[good] if good else None


# ── § Optimization ────────────────────────────────────────────────────────────
def _perf(w, mu, cov):
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def optimize(mu, cov, rf):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    sum1 = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    w0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        r, v = _perf(w, mu, cov)
        return -(r - rf) / v if v > 1e-12 else 1e9

    w_sharpe = minimize(neg_sharpe, w0, method="SLSQP",
                        bounds=bounds, constraints=[sum1]).x
    w_minvar = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                        bounds=bounds, constraints=[sum1]).x

    # Efficient frontier: min variance across a grid of target returns.
    # Only the UPPER branch (returns >= the min-variance portfolio's return) is
    # efficient; the lower branch is dominated, so we start the grid there to
    # avoid plotting the inefficient half.
    r_minvar = _perf(w_minvar, mu, cov)[0]
    frontier = []
    for tgt in np.linspace(r_minvar, float(mu.max()), 34):
        cons = [sum1, {"type": "eq", "fun": lambda w, t=tgt: float(w @ mu) - t}]
        r = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                     bounds=bounds, constraints=cons)
        if r.success:
            ret, vol = _perf(r.x, mu, cov)
            frontier.append((vol, ret))
    frontier.sort(key=lambda p: p[1])   # order by return along the efficient branch
    return w_sharpe, w_minvar, frontier


# ── § Charts ──────────────────────────────────────────────────────────────────
def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_frontier(tickers, mu, cov, frontier, w_sharpe, w_minvar, rf, cur_w=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(S1)
    ax.tick_params(colors=SUB, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BD)
    ax.grid(color=BD, lw=0.5, alpha=0.5)

    if frontier:
        fv = [f[0]*100 for f in frontier]; fr = [f[1]*100 for f in frontier]
        ax.plot(fv, fr, color=LINE1, lw=2.0, label="Efficient frontier", zorder=3)

    # Individual assets
    for i, t in enumerate(tickers):
        v = np.sqrt(cov[i, i]) * 100
        r = mu[i] * 100
        ax.scatter([v], [r], color=SUB, s=26, zorder=4)
        ax.annotate(t, (v, r), color=SUB, fontsize=7.5, xytext=(4, 2),
                    textcoords="offset points")

    for w, color, label, marker in [
        (w_sharpe, GRN,  "Max Sharpe", "*"),
        (w_minvar, LINE3, "Min variance", "D")]:
        r, v = _perf(w, mu, cov)
        ax.scatter([v*100], [r*100], color=color, s=(190 if marker == "*" else 90),
                   marker=marker, zorder=6, label=label, edgecolor=BG, linewidth=0.6)

    ew = np.repeat(1.0/len(tickers), len(tickers))
    r, v = _perf(ew, mu, cov)
    ax.scatter([v*100], [r*100], color=LINE2, s=70, marker="s", zorder=5,
               label="Equal weight", edgecolor=BG, linewidth=0.6)

    if cur_w is not None:
        r, v = _perf(cur_w, mu, cov)
        ax.scatter([v*100], [r*100], color=RED, s=90, marker="X", zorder=6,
                   label="Current", edgecolor=BG, linewidth=0.6)

    ax.set_xlabel("Annualised volatility (%)", color=SUB, fontsize=9)
    ax.set_ylabel("Annualised return (%)", color=SUB, fontsize=9)
    ax.set_title("Efficient Frontier", color=TEXT, fontsize=11, loc="left", pad=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD, labelcolor=SUB, loc="best")
    plt.tight_layout(pad=1.1)
    return _b64(fig)


def chart_corr(tickers, corr):
    n = len(tickers)
    fig, ax = plt.subplots(figsize=(max(4.5, n*0.55), max(4.0, n*0.5)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(S1)
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", color=SUB, fontsize=8)
    ax.set_yticklabels(tickers, color=SUB, fontsize=8)
    # RdYlGn_r is light across most of its range, so dark text reads best on
    # every cell (including the deep-red diagonal).
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="#1a1a1a", fontsize=7.5, fontweight="medium")
    ax.set_title("Return Correlation", color=TEXT, fontsize=11, loc="left", pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.ax.tick_params(colors=SUB, labelsize=7)
    plt.tight_layout(pad=1.1)
    return _b64(fig)


# ── § HTML ────────────────────────────────────────────────────────────────────
def _weights_table(tickers, portfolios):
    head = "".join(f"<th class='r'>{name}</th>" for name, _ in portfolios)
    rows = ""
    for i, t in enumerate(tickers):
        cells = ""
        for _, w in portfolios:
            wt = w[i] * 100
            shade = min(0.35, wt / 100 * 0.7)
            cells += f'<td class="r" style="background:rgba(99,102,241,{shade:.2f})">{wt:.1f}%</td>'
        rows += f'<tr><td class="tk">{t}</td>{cells}</tr>'
    return f"""<table><thead><tr><th>Ticker</th>{head}</tr></thead><tbody>{rows}</tbody></table>"""


def build_html(tickers, mu, cov, corr, frontier, w_sharpe, w_minvar, rf, risk, ts):
    ew = np.repeat(1.0/len(tickers), len(tickers))
    portfolios = [("Max Sharpe", w_sharpe), ("Min Var", w_minvar), ("Equal Wt", ew)]

    def stat(w):
        r, v = _perf(w, mu, cov)
        sh = (r - rf) / v if v > 0 else 0
        return r*100, v*100, sh
    ms_r, ms_v, ms_s = stat(w_sharpe)

    risk_html = ""
    if risk:
        risk_html = f"""
<div class="cards">
  <div class="card"><div class="cv">{risk.get('var_95','--')}%</div><div class="cl">1-day VaR 95%</div></div>
  <div class="card"><div class="cv">{risk.get('max_drawdown','--')}%</div><div class="cl">Max drawdown</div></div>
  <div class="card"><div class="cv">{risk.get('calmar_ratio','--')}</div><div class="cl">Calmar ratio</div></div>
  <div class="card"><div class="cv">{risk.get('annual_return','--')}%</div><div class="cl">Ann. return (hist)</div></div>
</div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Optimizer &mdash; {ts}</title><style>
:root{{--bg:{BG};--s1:{S1};--bd:{BD};--text:{TEXT};--sub:{SUB}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:{BG};color:{TEXT};padding-bottom:60px}}
.hero{{padding:26px 30px 20px;background:linear-gradient(135deg,{BG},{S1} 70%);border-bottom:1px solid {BD}}}
.hero h1{{font-size:21px;font-weight:800}}.hero .sub{{font-size:12px;color:{SUB};margin-top:4px}}
.stats{{display:flex;gap:30px;margin-top:16px;flex-wrap:wrap}}
.sv{{font-size:20px;font-weight:800;color:{GRN}}}.sl{{font-size:10px;color:{SUB};text-transform:uppercase;letter-spacing:1px}}
.wrap{{padding:22px 30px;display:flex;flex-direction:column;gap:22px;max-width:1100px}}
.row{{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}}
.panel{{background:{S1};border:1px solid {BD};border-radius:12px;padding:14px}}
.panel img{{max-width:100%;display:block}}
h2{{font-size:13px;margin-bottom:10px;color:{TEXT}}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{padding:6px 12px;border-bottom:1px solid {BD};text-align:left}}
th{{color:{SUB};font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}}
td.r,th.r{{text-align:right}}td.tk{{font-weight:700}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}}
.card{{background:{S2};border:1px solid {BD};border-radius:10px;padding:10px 16px;min-width:120px}}
.cv{{font-size:17px;font-weight:800;color:{TEXT}}}.cl{{font-size:10px;color:{SUB};text-transform:uppercase;letter-spacing:.5px}}
.note{{font-size:11px;color:{SUB};line-height:1.5}}
.footer{{padding:16px 30px;border-top:1px solid {BD};font-size:11px;color:{SUB}}}
</style></head><body>
<div class="hero"><h1>Portfolio Optimizer &bull; Efficient Frontier</h1>
<div class="sub">Generated {ts} &bull; {len(tickers)} assets &bull; Long-only mean-variance optimization</div>
<div class="stats">
  <div><div class="sv">{ms_r:.1f}%</div><div class="sl">Max-Sharpe return</div></div>
  <div><div class="sv">{ms_v:.1f}%</div><div class="sl">Max-Sharpe volatility</div></div>
  <div><div class="sv">{ms_s:.2f}</div><div class="sl">Sharpe ratio</div></div>
</div></div>
<div class="wrap">
  <div class="row">
    <div class="panel"><img src="data:image/png;base64,{chart_frontier(tickers, mu, cov, frontier, w_sharpe, w_minvar, rf)}"></div>
    <div class="panel"><img src="data:image/png;base64,{chart_corr(tickers, corr)}"></div>
  </div>
  <div class="panel">
    <h2>Recommended portfolio (max Sharpe) &mdash; risk profile</h2>
    {risk_html}
  </div>
  <div class="panel">
    <h2>Optimal weights</h2>
    {_weights_table(tickers, portfolios)}
    <div class="note" style="margin-top:10px">Cells shaded by weight. Max-Sharpe maximises return per unit of
    risk; Min-Var minimises volatility; Equal-Wt is the naive baseline.</div>
  </div>
</div>
<div class="footer">Data: yfinance (~3y daily). Historical mean returns are noisy -- use the frontier
shape and correlations, which are more stable than the point estimates. Not investment advice.</div>
</body></html>"""


# ── § Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Portfolio Optimizer -- Efficient Frontier & Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("tickers", nargs="?", default=None, help="Comma-separated tickers")
    ap.add_argument("--tickers", dest="tickers_flag", default=None)
    ap.add_argument("--years", type=int, default=3, help="Years of history (default 3)")
    ap.add_argument("--rf", type=float, default=0.043, help="Annual risk-free rate (default 0.043)")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    raw = args.tickers or args.tickers_flag
    if not raw:
        print("ERROR: provide tickers, e.g. portfolioOptimizer.py NVDA,MSFT,AAPL,GOOGL")
        sys.exit(1)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if len(tickers) < 2:
        print("ERROR: need at least 2 tickers to optimize.")
        sys.exit(1)

    print("\n" + "=" * 58)
    print("  PORTFOLIO OPTIMIZER -- EFFICIENT FRONTIER")
    print("=" * 58)
    print(f"\n  Fetching {args.years}y price history for {len(tickers)} tickers...")
    prices = fetch_prices(tickers, args.years)
    if prices is None or prices.shape[1] < 2:
        print("  FAILED: not enough overlapping price history.")
        sys.exit(1)
    tickers = list(prices.columns)
    print(f"  {prices.shape[0]} trading days, {len(tickers)} usable tickers.")

    rets = prices.pct_change().dropna()
    mu   = rets.mean().values * 252
    cov  = rets.cov().values * 252
    corr = rets.corr().values

    w_sharpe, w_minvar, frontier = optimize(mu, cov, args.rf)

    # Risk profile of the recommended (max-Sharpe) portfolio
    risk = None
    if calc_portfolio_risk is not None:
        weights = {t: float(w_sharpe[i]) for i, t in enumerate(tickers)}
        histories = {t: prices[t] for t in tickers}
        try:
            risk = calc_portfolio_risk(weights, histories, rf_annual=args.rf)
        except Exception:
            risk = None

    # Console summary
    print("\n  Max-Sharpe weights:")
    order = np.argsort(w_sharpe)[::-1]
    for i in order:
        if w_sharpe[i] > 0.005:
            print(f"    {tickers[i]:<7} {w_sharpe[i]*100:5.1f}%")
    r, v = _perf(w_sharpe, mu, cov)
    print(f"  -> return {r*100:.1f}%  vol {v*100:.1f}%  Sharpe {(r-args.rf)/v:.2f}")
    if risk:
        print(f"  -> 1-day VaR95 {risk.get('var_95')}%  max drawdown {risk.get('max_drawdown')}%")

    ts = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolioOptimizerData")
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"portfolio_optimizer_{datetime.datetime.now():%Y%m%d_%H%M}.html")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(build_html(tickers, mu, cov, corr, frontier, w_sharpe, w_minvar, args.rf, risk, ts))
    print(f"\n  HTML saved: {fpath}  ({os.path.getsize(fpath)//1024} KB)")
    if args.open:
        webbrowser.open("file://" + fpath)


if __name__ == "__main__":
    main()
