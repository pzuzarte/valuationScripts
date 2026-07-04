"""
Options-Implied Risk Dashboard
==============================
Extracts forward-looking risk signals from live option chains (yfinance) for
one or more tickers.  Where price history tells you what *has* happened, the
options market prices what traders think *will* happen.

Metrics (per ticker)
--------------------
  * ATM implied volatility  -- 30-day at-the-money IV (the market's expected move)
  * Put/call skew           -- IV(10% OTM put) - IV(10% OTM call).  Positive =
                               downside protection is bid up = crash fear.
  * Term-structure slope    -- far-dated ATM IV minus near-dated ATM IV.
                               Negative (backwardation) = acute near-term stress.
  * Put/call OI ratio       -- open interest in puts vs calls (positioning).
  * Realized volatility     -- 21-day annualised, from price history.
  * Variance risk premium   -- ATM IV minus realized vol.  Positive = options
                               are 'expensive' vs recent movement (typical).

USAGE
-----
  python optionsRisk.py NVDA
  python optionsRisk.py NVDA,AAPL,TSLA,SPY --open
  python optionsRisk.py --tickers MSFT,GOOGL

Notes: true IV rank needs a history of implied vol (not freely available), so
this reports the current IV *level* plus its variance-risk-premium context
rather than a percentile rank.
"""

import sys, os, io, argparse, base64, datetime, webbrowser, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

BG = "#0f1117"; S1 = "#161a27"; S2 = "#1a1e2e"; BD = "#252a3a"
TEXT = "#e2e8f0"; SUB = "#94a3b8"
GRN = "#22c55e"; LINE1 = "#6366f1"; LINE2 = "#14b8a6"; LINE3 = "#f59e0b"
RED = "#ef4444"; YLW = "#eab308"; ORANGE = "#f59e0b"


def _clean_chain(df):
    """Keep rows with a sane IV and some liquidity."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    df = df[(df["impliedVolatility"] > 0.02) & (df["impliedVolatility"] < 5.0)]
    df = df[df["strike"] > 0]
    return df


def _iv_at_strike(df, target_strike):
    """Linear-interpolate IV at target_strike from a cleaned chain (by strike)."""
    if df is None or len(df) < 2:
        return None
    d = df.sort_values("strike")
    strikes = d["strike"].values
    ivs     = d["impliedVolatility"].values
    if target_strike <= strikes[0]:
        return float(ivs[0])
    if target_strike >= strikes[-1]:
        return float(ivs[-1])
    return float(np.interp(target_strike, strikes, ivs))


def _atm_iv(calls, puts, spot):
    """Average of the call and put IV nearest the spot (ATM)."""
    ivs = []
    for d in (calls, puts):
        if d is not None and len(d):
            i = (d["strike"] - spot).abs().idxmin()
            ivs.append(float(d.loc[i, "impliedVolatility"]))
    return float(np.mean(ivs)) if ivs else None


def _realized_vol(tk, window=21):
    """Annualised realized volatility from daily log returns."""
    try:
        h = tk.history(period="3mo")
        if h is None or len(h) < window + 2:
            return None
        r = np.log(h["Close"] / h["Close"].shift(1)).dropna()
        return float(r.tail(window).std() * math.sqrt(252))
    except Exception:
        return None


def fetch_option_metrics(ticker):
    """Compute the options-risk metric bundle for one ticker."""
    tk = yf.Ticker(ticker)
    try:
        spot = float(tk.history(period="1d")["Close"].iloc[-1])
    except Exception:
        return None
    exps = list(tk.options or [])
    if not exps:
        return None

    today = datetime.date.today()
    dated = []
    for e in exps:
        try:
            d = (datetime.date.fromisoformat(e) - today).days
            if d > 3:
                dated.append((d, e))
        except Exception:
            pass
    if not dated:
        return None
    dated.sort()

    def _closest(target):
        return min(dated, key=lambda x: abs(x[0] - target))

    near_days, near_exp = _closest(30)
    far_days,  far_exp  = _closest(90)

    term = []
    for tgt in (30, 60, 90, 180):
        dd, ee = _closest(tgt)
        try:
            ch = tk.option_chain(ee)
            iv = _atm_iv(_clean_chain(ch.calls), _clean_chain(ch.puts), spot)
            if iv:
                term.append((dd, iv))
        except Exception:
            pass
    # de-dup by days
    term = sorted({d: v for d, v in term}.items())

    # Near-expiry chain for ATM IV, skew, smile, put/call OI
    ch = tk.option_chain(near_exp)
    calls, puts = _clean_chain(ch.calls), _clean_chain(ch.puts)
    atm_iv = _atm_iv(calls, puts, spot)

    iv_put_10  = _iv_at_strike(puts,  spot * 0.90)   # 10% OTM put
    iv_call_10 = _iv_at_strike(calls, spot * 1.10)   # 10% OTM call
    skew = (iv_put_10 - iv_call_10) if (iv_put_10 and iv_call_10) else None

    put_oi  = float(puts["openInterest"].fillna(0).sum())  if puts  is not None and len(puts)  else 0.0
    call_oi = float(calls["openInterest"].fillna(0).sum()) if calls is not None and len(calls) else 0.0
    pc_oi   = (put_oi / call_oi) if call_oi > 0 else None

    term_slope = None
    if len(term) >= 2:
        term_slope = term[-1][1] - term[0][1]

    rv  = _realized_vol(tk)
    vrp = (atm_iv - rv) if (atm_iv and rv) else None

    # Smile data (near expiry): IV vs moneyness, OTM puts + OTM calls
    smile = []
    if puts is not None:
        for _, row in puts.iterrows():
            if row["strike"] <= spot:
                smile.append((row["strike"] / spot, float(row["impliedVolatility"]), "put"))
    if calls is not None:
        for _, row in calls.iterrows():
            if row["strike"] >= spot:
                smile.append((row["strike"] / spot, float(row["impliedVolatility"]), "call"))
    smile.sort()

    return {
        "ticker": ticker.upper(), "spot": spot,
        "atm_iv": atm_iv, "skew": skew, "term_slope": term_slope,
        "pc_oi": pc_oi, "realized_vol": rv, "vrp": vrp,
        "near_days": near_days, "far_days": far_days,
        "term": term, "smile": smile,
        "iv_put_10": iv_put_10, "iv_call_10": iv_call_10,
    }


# ── § Charts ──────────────────────────────────────────────────────────────────
def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_smile_term(m):
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax in (axs, axt):
        fig.patch.set_facecolor(BG); ax.set_facecolor(S1)
        ax.tick_params(colors=SUB, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BD)
        ax.grid(color=BD, lw=0.5, alpha=0.5)

    # Smile
    smile = m["smile"]
    if smile:
        puts  = [(x*100, y*100) for x, y, s in smile if s == "put"]
        calls = [(x*100, y*100) for x, y, s in smile if s == "call"]
        if puts:
            axs.plot([p[0] for p in puts], [p[1] for p in puts], color=RED, lw=1.6, label="OTM puts")
        if calls:
            axs.plot([c[0] for c in calls], [c[1] for c in calls], color=LINE2, lw=1.6, label="OTM calls")
        axs.axvline(100, color=SUB, ls="--", lw=0.8, alpha=0.6)
        axs.legend(fontsize=8, framealpha=0.2, facecolor=S2, edgecolor=BD, labelcolor=SUB)
    axs.set_title(f"{m['ticker']}  Volatility Smile ({m['near_days']}d)",
                  color=TEXT, fontsize=10, loc="left")
    axs.set_xlabel("Strike / Spot (%)", color=SUB, fontsize=8)
    axs.set_ylabel("Implied vol (%)", color=SUB, fontsize=8)

    # Term structure
    term = m["term"]
    if len(term) >= 2:
        axt.plot([t[0] for t in term], [t[1]*100 for t in term],
                 color=LINE3, lw=1.8, marker="o", ms=4)
    axt.set_title("ATM IV Term Structure", color=TEXT, fontsize=10, loc="left")
    axt.set_xlabel("Days to expiry", color=SUB, fontsize=8)
    axt.set_ylabel("ATM IV (%)", color=SUB, fontsize=8)

    plt.tight_layout(pad=1.2)
    return _b64(fig)


# ── § HTML ────────────────────────────────────────────────────────────────────
def _pct(v, dp=1):
    return f"{v*100:.{dp}f}%" if isinstance(v, (int, float)) else "--"


def _interpret(m):
    notes = []
    if m["skew"] is not None:
        if m["skew"] > 0.06:   notes.append(("Elevated crash fear (steep put skew)", RED))
        elif m["skew"] > 0.02: notes.append(("Normal downside skew", SUB))
        else:                  notes.append(("Flat/low skew -- complacent", YLW))
    if m["term_slope"] is not None and m["term_slope"] < -0.01:
        notes.append(("Backwardated term structure -- near-term stress", RED))
    if m["vrp"] is not None:
        if m["vrp"] > 0.03:    notes.append(("Options rich vs realized (VRP+)", LINE2))
        elif m["vrp"] < -0.02: notes.append(("Options cheap vs realized (VRP-)", ORANGE))
    if m["pc_oi"] is not None and m["pc_oi"] > 1.3:
        notes.append(("Heavy put positioning", YLW))
    return notes


def build_html(metrics, ts, open_it=False):
    # comparison table
    trows = ""
    for m in metrics:
        sk_c = RED if (m["skew"] and m["skew"] > 0.06) else (
            YLW if (m["skew"] is not None and m["skew"] < 0.02) else TEXT)
        ts_c = RED if (m["term_slope"] is not None and m["term_slope"] < -0.01) else TEXT
        trows += f"""
<tr>
  <td class="tk">{m['ticker']}</td>
  <td class="r">${m['spot']:,.2f}</td>
  <td class="r" style="font-weight:700">{_pct(m['atm_iv'])}</td>
  <td class="r" style="color:{sk_c}">{_pct(m['skew'],1) if m['skew'] is not None else '--'}</td>
  <td class="r" style="color:{ts_c}">{(_pct(m['term_slope'],1) if m['term_slope'] is not None else '--')}</td>
  <td class="r">{f"{m['pc_oi']:.2f}" if m['pc_oi'] is not None else '--'}</td>
  <td class="r">{_pct(m['realized_vol'])}</td>
  <td class="r">{_pct(m['vrp'],1) if m['vrp'] is not None else '--'}</td>
</tr>"""

    detail = ""
    for m in metrics:
        notes = _interpret(m)
        chips = "".join(
            f'<span class="chip" style="color:{c};border-color:{c}55;background:{c}18">{t}</span>'
            for t, c in notes)
        detail += f"""
<div class="section">
  <div class="sec-hdr"><span class="sec-title">{m['ticker']} &bull; ${m['spot']:,.2f}</span>
    <span class="chips">{chips}</span></div>
  <img class="chart" src="data:image/png;base64,{chart_smile_term(m)}" alt="{m['ticker']} options charts">
</div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Options-Implied Risk &mdash; {ts}</title><style>
:root{{--bg:{BG};--s1:{S1};--bd:{BD};--text:{TEXT};--sub:{SUB}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:{BG};color:{TEXT};padding-bottom:60px}}
.hero{{padding:26px 30px 20px;background:linear-gradient(135deg,{BG},{S1} 70%);border-bottom:1px solid {BD}}}
.hero h1{{font-size:21px;font-weight:800}}.hero .sub{{font-size:12px;color:{SUB};margin-top:4px}}
.wrap{{padding:22px 30px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:26px}}
th,td{{padding:7px 10px;border-bottom:1px solid {BD};text-align:left;white-space:nowrap}}
th{{color:{SUB};font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}}
td.r,th.r{{text-align:right}}td.tk{{font-weight:700}}
.section{{background:{S1};border:1px solid {BD};border-radius:12px;overflow:hidden;margin-bottom:20px}}
.sec-hdr{{padding:12px 18px;border-bottom:1px solid {BD};display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.sec-title{{font-size:13px;font-weight:700}}
.chips{{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}}
.chip{{font-size:10px;padding:2px 8px;border-radius:10px;border:1px solid}}
.chart{{width:100%;display:block}}
.legend{{font-size:11px;color:{SUB};line-height:1.6;margin-top:6px}}
.footer{{padding:16px 30px;border-top:1px solid {BD};font-size:11px;color:{SUB}}}
</style></head><body>
<div class="hero"><h1>Options-Implied Risk Dashboard</h1>
<div class="sub">Generated {ts} &bull; Forward-looking risk from live option chains</div></div>
<div class="wrap">
<table><thead><tr>
  <th>Ticker</th><th class="r">Spot</th><th class="r">ATM IV</th><th class="r">Put-Call Skew</th>
  <th class="r">Term Slope</th><th class="r">P/C OI</th><th class="r">Realized Vol</th><th class="r">VRP</th>
</tr></thead><tbody>{trows}</tbody></table>
<div class="legend">
  <b>ATM IV</b>: 30-day at-the-money implied vol (expected move). &bull;
  <b>Skew</b>: IV(10% OTM put) - IV(10% OTM call); higher = more crash fear. &bull;
  <b>Term slope</b>: 180d ATM IV - 30d; negative = near-term stress (backwardation). &bull;
  <b>P/C OI</b>: put vs call open interest. &bull;
  <b>VRP</b>: ATM IV - realized vol; positive = options 'expensive' vs recent moves.
</div>
{detail}
</div>
<div class="footer">Data: yfinance option chains &bull; True IV-rank needs historical IV (not free);
this shows current IV level + variance-risk-premium context. &bull; Not investment advice.</div>
</body></html>"""


# ── § Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Options-Implied Risk Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("tickers", nargs="?", default=None,
                    help="Comma-separated tickers (e.g. NVDA,AAPL,SPY)")
    ap.add_argument("--tickers", dest="tickers_flag", default=None,
                    help="Alternative way to pass comma-separated tickers")
    ap.add_argument("--open", action="store_true", help="Open the HTML report in a browser")
    args = ap.parse_args()

    raw = args.tickers or args.tickers_flag
    if not raw:
        print("ERROR: provide one or more tickers, e.g. optionsRisk.py NVDA,AAPL,SPY")
        sys.exit(1)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    print("\n" + "=" * 58)
    print("  OPTIONS-IMPLIED RISK DASHBOARD")
    print("=" * 58)

    metrics = []
    for t in tickers:
        print(f"\n  Fetching option chain for {t}...")
        try:
            m = fetch_option_metrics(t)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        if m is None:
            print(f"    No option data for {t} -- skipped.")
            continue
        metrics.append(m)
        print("    ATM IV {:>6}  skew {:>6}  term {:>6}  P/C {:>5}  realized {:>6}  VRP {:>6}".format(
            _pct(m["atm_iv"]),
            (_pct(m["skew"], 1) if m["skew"] is not None else "--"),
            (_pct(m["term_slope"], 1) if m["term_slope"] is not None else "--"),
            (f"{m['pc_oi']:.2f}" if m["pc_oi"] is not None else "--"),
            _pct(m["realized_vol"]),
            (_pct(m["vrp"], 1) if m["vrp"] is not None else "--")))

    if not metrics:
        print("\n  No option data retrieved for any ticker.")
        sys.exit(1)

    ts = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optionsRiskData")
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"options_risk_{datetime.datetime.now():%Y%m%d_%H%M}.html")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(build_html(metrics, ts))
    print(f"\n  HTML saved: {fpath}  ({os.path.getsize(fpath)//1024} KB)")
    if args.open:
        webbrowser.open("file://" + fpath)


if __name__ == "__main__":
    main()
