"""
Earnings Quality & Accruals Screener
====================================
Ranks a universe of large-cap US stocks by *earnings quality* -- how well a
company's reported profits are backed by real cash, using accrual analysis.

Why it matters
--------------
Reported net income can diverge from cash generation through accounting
accruals (revenue booked before cash arrives, capitalised costs, working-capital
swings).  Sloan (1996) showed that firms with high accruals -- earnings NOT
backed by cash -- systematically underperform.  This tool surfaces that signal.

Metrics (all TTM, from the TradingView screener)
------------------------------------------------
  * Sloan accruals ratio = (Net Income - Operating Cash Flow) / Total Assets
        Lower / negative is better (earnings are conservative, cash-backed).
        High positive is a red flag (earnings run ahead of cash).
  * Cash conversion  = Operating Cash Flow / Net Income     (>1 is strong)
  * FCF conversion   = Free Cash Flow / Net Income          (>0.8 is strong)
  * Balance-sheet support = current ratio + debt/equity

Composite Earnings-Quality score (0-100):
    Accruals (40) + Cash conversion (30) + FCF conversion (20) + Balance (10)
Loss-making firms are scored on cash generation only and capped (the accruals
model is not meaningful when net income is negative).

USAGE
-----
  python earningsQuality.py                       # top 500 US large caps
  python earningsQuality.py --limit 300
  python earningsQuality.py --min-mktcap 5e9 --open

Financials (banks / insurers) are included but flagged: (NI - CFO)/assets is
distorted for lending businesses, so treat their scores with caution.
"""

import sys, os, io, argparse, base64, datetime, webbrowser
import pandas as pd

# ── Colours (match the suite dark theme) ──────────────────────────────────────
BG   = "#0f1117"; S1 = "#161a27"; S2 = "#1a1e2e"; BD = "#252a3a"
TEXT = "#e2e8f0"; SUB = "#94a3b8"
GRN  = "#22c55e"; LINE2 = "#14b8a6"; YLW = "#eab308"; ORANGE = "#f59e0b"; RED = "#ef4444"

# ── TradingView fields (all validated to return data) ─────────────────────────
FIELDS = [
    "name", "description", "close", "market_cap_basic", "sector",
    "net_income", "free_cash_flow", "cash_f_operating_activities_ttm",
    "total_assets", "total_revenue",
    "gross_margin", "after_tax_margin", "return_on_assets",
    "debt_to_equity", "current_ratio",
]


# ── § Universe fetch ──────────────────────────────────────────────────────────
def fetch_universe(limit=500, min_mktcap=2e9):
    """Top-N US primary common stocks by market cap, via TradingView."""
    from tradingview_screener import Query, col
    q = (Query()
         .select(*FIELDS)
         .set_markets("america")
         .where(col("market_cap_basic") > min_mktcap,
                col("is_primary") == True)                       # noqa: E712
         .order_by("market_cap_basic", ascending=False)
         .limit(int(limit)))
    _, df = q.get_scanner_data()
    df = df.dropna(subset=["market_cap_basic", "net_income", "total_assets"])
    for c in FIELDS:
        if c not in ("name", "description", "sector"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


# ── § Scoring components ──────────────────────────────────────────────────────
def _score_accruals(ar):
    """(NI - CFO)/TA -> 0-40.  Lower accruals = higher score.
    Top band is deliberately hard (needs accruals <= -10% of assets) so
    genuinely elite cash generators separate from the merely-good."""
    if ar is None:
        return None
    if ar <= -0.10: return 40.0
    if ar <= -0.05: return 34.0 + ((-0.05 - ar) / 0.05) * 6.0
    if ar <=  0.00: return 28.0 + ((-ar) / 0.05) * 6.0
    if ar <=  0.05: return 16.0 + ((0.05 - ar) / 0.05) * 12.0
    if ar <=  0.10: return  8.0 + ((0.10 - ar) / 0.05) * 8.0
    return max(0.0, 8.0 - (ar - 0.10) * 40.0)


def _score_cashconv(cc):
    """CFO/NI -> 0-30.  Full marks require CFO >= 1.5x net income."""
    if cc is None:
        return None
    if cc >= 1.5: return 30.0
    if cc >= 1.2: return 26.0 + (cc - 1.2) / 0.3 * 4.0
    if cc >= 1.0: return 22.0 + (cc - 1.0) / 0.2 * 4.0
    if cc >= 0.8: return 15.0 + (cc - 0.8) / 0.2 * 7.0
    if cc >= 0.5: return  7.0 + (cc - 0.5) / 0.3 * 8.0
    if cc >= 0.0: return        (cc / 0.5) * 7.0
    return 0.0


def _score_fcfconv(fc):
    """FCF/NI -> 0-20.  Full marks require FCF >= 1.2x net income."""
    if fc is None:
        return None
    if fc >= 1.2: return 20.0
    if fc >= 1.0: return 17.0 + (fc - 1.0) / 0.2 * 3.0
    if fc >= 0.8: return 13.0 + (fc - 0.8) / 0.2 * 4.0
    if fc >= 0.5: return  7.0 + (fc - 0.5) / 0.3 * 6.0
    if fc >= 0.0: return        (fc / 0.5) * 7.0
    return 0.0


def _score_balance(cr, de):
    """Current ratio + debt/equity -> 0-10."""
    s = 0.0
    if cr is not None:
        if   cr >= 1.5: s += 5.0
        elif cr >= 1.0: s += 2.5 + (cr - 1.0) / 0.5 * 2.5
        elif cr >= 0.5: s += (cr - 0.5) / 0.5 * 2.5
    else:
        s += 2.5
    if de is not None:
        if   de <= 1.0: s += 5.0
        elif de <= 2.0: s += 5.0 - (de - 1.0) / 1.0 * 5.0
    else:
        s += 2.5
    return s


def _tier(score):
    if score >= 80: return "Excellent", GRN
    if score >= 65: return "Good",      LINE2
    if score >= 50: return "Fair",      YLW
    if score >= 35: return "Weak",      ORANGE
    return "Poor", RED


def score_row(r):
    """Compute earnings-quality metrics + composite score for one screener row."""
    def g(f):
        v = r.get(f)
        try:
            v = float(v)
            return v if v == v else None       # drop NaN
        except (TypeError, ValueError):
            return None

    ni   = g("net_income")
    cfo  = g("cash_f_operating_activities_ttm")
    fcf  = g("free_cash_flow")
    ta   = g("total_assets")
    cr   = g("current_ratio")
    de   = g("debt_to_equity")
    sector = str(r.get("sector") or "")

    if ni is None or ta is None or ta <= 0:
        return None

    accruals = ((ni - cfo) / ta) if cfo is not None else None
    cc = (cfo / ni) if (cfo is not None and ni > 0) else None
    fc = (fcf / ni) if (fcf is not None and ni > 0) else None

    flags = []
    if ni > 0:
        s_acc = _score_accruals(accruals) if accruals is not None else 20.0
        s_cc  = _score_cashconv(cc)       if cc       is not None else 15.0
        s_fc  = _score_fcfconv(fc)        if fc       is not None else 10.0
        s_bal = _score_balance(cr, de)
        score = s_acc + s_cc + s_fc + s_bal

        if cfo is not None and cfo < 0:
            flags.append("Positive earnings but NEGATIVE operating cash flow")
        if fcf is not None and fcf < 0:
            flags.append("Positive earnings but negative FCF")
        if accruals is not None and accruals > 0.10:
            flags.append("High accruals -- earnings not cash-backed")
        if cc is not None and cc < 0.6:
            flags.append("Weak cash conversion (CFO < 60% of NI)")
    else:
        # Loss-maker: cash-conversion ratios undefined; score cash generation only.
        cash_pos = (cfo is not None and cfo > 0)
        fcf_pos  = (fcf is not None and fcf > 0)
        base = 45.0 if (cash_pos and fcf_pos) else (28.0 if (cash_pos or fcf_pos) else 8.0)
        score = base + _score_balance(cr, de)      # capped ~55
        flags.append("Net loss (TTM) -- accruals model N/A")

    if de is not None and de > 3.0:
        flags.append("High leverage (D/E > 3)")
    if sector == "Finance":
        flags.append("Financial -- accruals metric less meaningful")

    tier, color = _tier(score)
    return {
        "ticker":     str(r.get("name", "?")),
        "company":    str(r.get("description", "") or "").strip(),
        "sector":     sector or "--",
        "price":      g("close"),
        "mktcap_bn":  (g("market_cap_basic") or 0) / 1e9,
        "score":      round(score, 1),
        "tier":       tier,
        "color":      color,
        "accruals":   round(accruals * 100, 2) if accruals is not None else None,
        "cash_conv":  round(cc, 2) if cc is not None else None,
        "fcf_conv":   round(fc, 2) if fc is not None else None,
        "gross_margin": g("gross_margin"),
        "net_margin":   g("after_tax_margin"),
        "roa":          g("return_on_assets"),
        "de":           de,
        "flags":      flags,
    }


# ── § HTML report ─────────────────────────────────────────────────────────────
def _fmt(v, suffix="", dp=1):
    return f"{v:.{dp}f}{suffix}" if isinstance(v, (int, float)) else "--"


def build_html(rows, ts, n_universe, suite_port=5050):
    body = []
    for i, r in enumerate(rows, 1):
        flag_html = ""
        if r["flags"]:
            chips = "".join(
                f'<span class="flag">{f}</span>' for f in r["flags"])
            flag_html = f'<div class="flags">{chips}</div>'
        acc = r["accruals"]
        acc_color = RED if (acc is not None and acc > 10) else (
            GRN if (acc is not None and acc < 0) else SUB)
        body.append(f"""
<tr>
  <td class="num">{i}</td>
  <td class="tk">{r['ticker']}</td>
  <td class="co">{r['company'][:34]}</td>
  <td class="sec">{r['sector']}</td>
  <td class="r"><span class="score" style="background:{r['color']}22;color:{r['color']};border:1px solid {r['color']}55">{r['score']:.0f}</span></td>
  <td class="r" style="color:{r['color']};font-weight:600">{r['tier']}</td>
  <td class="r" style="color:{acc_color}">{_fmt(r['accruals'],'%',2)}</td>
  <td class="r">{_fmt(r['cash_conv'],'x',2)}</td>
  <td class="r">{_fmt(r['fcf_conv'],'x',2)}</td>
  <td class="r">{_fmt(r['net_margin'],'%')}</td>
  <td class="r">{_fmt(r['roa'],'%')}</td>
  <td class="r">{_fmt(r['mktcap_bn'],'B',0)}</td>
  <td>{flag_html}</td>
</tr>""")

    rows_html = "".join(body)
    n_excellent = sum(1 for r in rows if r["score"] >= 80)
    n_poor      = sum(1 for r in rows if r["score"] < 35)
    n_redflag   = sum(1 for r in rows if any("NEGATIVE" in f or "High accruals" in f for f in r["flags"]))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earnings Quality &amp; Accruals Screener &mdash; {ts}</title>
<style>
:root{{--bg:{BG};--s1:{S1};--s2:{S2};--bd:{BD};--text:{TEXT};--sub:{SUB}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text)}}
.hero{{padding:26px 30px 20px;background:linear-gradient(135deg,{BG},{S1} 60%,{S2});border-bottom:1px solid var(--bd)}}
.hero h1{{font-size:21px;font-weight:800}}
.hero .sub{{font-size:12px;color:var(--sub);margin-top:4px}}
.stats{{display:flex;gap:26px;margin-top:16px;flex-wrap:wrap}}
.stat-val{{font-size:20px;font-weight:800}}
.stat-lbl{{font-size:10px;color:var(--sub);text-transform:uppercase;letter-spacing:1px}}
.wrap{{padding:22px 30px 60px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:var(--s1);color:var(--sub);font-size:10.5px;text-transform:uppercase;
    letter-spacing:.6px;cursor:pointer;user-select:none;z-index:5}}
th:hover{{color:var(--text)}}
td.r,th.r{{text-align:right}}
td.num{{color:var(--sub);font-size:11px}}
td.tk{{font-weight:700}}
td.co{{color:var(--sub);max-width:230px;overflow:hidden;text-overflow:ellipsis}}
td.sec{{color:var(--sub);font-size:11px}}
.score{{display:inline-block;min-width:34px;text-align:center;padding:2px 7px;border-radius:8px;font-weight:700}}
.flags{{display:flex;gap:5px;flex-wrap:wrap;max-width:340px;white-space:normal}}
.flag{{font-size:10px;background:rgba(239,68,68,.12);color:#fca5a5;border:1px solid rgba(239,68,68,.3);
       padding:1px 6px;border-radius:8px}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.footer{{padding:16px 30px;border-top:1px solid var(--bd);font-size:11px;color:var(--sub)}}
</style></head>
<body>
<div class="hero">
  <h1>Earnings Quality &amp; Accruals Screener</h1>
  <div class="sub">Generated {ts} &bull; {n_universe} US large caps screened &bull; Higher score = earnings better backed by cash</div>
  <div class="stats">
    <div><div class="stat-val" style="color:{GRN}">{n_excellent}</div><div class="stat-lbl">Excellent (80+)</div></div>
    <div><div class="stat-val" style="color:{ORANGE}">{n_poor}</div><div class="stat-lbl">Poor (&lt;35)</div></div>
    <div><div class="stat-val" style="color:{RED}">{n_redflag}</div><div class="stat-lbl">Cash-flow red flags</div></div>
  </div>
</div>
<div class="wrap">
<table id="t">
<thead><tr>
  <th>#</th><th onclick="sortT(1,'s')">Ticker</th><th onclick="sortT(2,'s')">Company</th>
  <th onclick="sortT(3,'s')">Sector</th><th class="r" onclick="sortT(4,'n')">Score</th>
  <th onclick="sortT(5,'s')">Tier</th><th class="r" onclick="sortT(6,'n')">Accruals</th>
  <th class="r" onclick="sortT(7,'n')">CFO/NI</th><th class="r" onclick="sortT(8,'n')">FCF/NI</th>
  <th class="r" onclick="sortT(9,'n')">Net Mgn</th><th class="r" onclick="sortT(10,'n')">ROA</th>
  <th class="r" onclick="sortT(11,'n')">Mkt Cap</th><th>Flags</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<div class="footer">
  Accruals ratio = (Net Income - Operating Cash Flow) / Total Assets (TTM).  Lower / negative is better.
  &bull; Data: TradingView.  &bull; Not investment advice.
</div>
<script>
let dir={{}};
function sortT(col,type){{
  const tb=document.querySelector('#t tbody');
  const rows=[...tb.rows];
  dir[col]=!dir[col];
  const s=dir[col]?1:-1;
  rows.sort((a,b)=>{{
    let x=a.cells[col].innerText.replace(/[%xB,]/g,'').trim();
    let y=b.cells[col].innerText.replace(/[%xB,]/g,'').trim();
    if(type==='n'){{x=parseFloat(x);y=parseFloat(y);
       if(isNaN(x))x=-1e9;if(isNaN(y))y=-1e9;return (x-y)*s;}}
    return x.localeCompare(y)*s;
  }});
  rows.forEach(r=>tb.appendChild(r));
}}
</script>
</body></html>"""


# ── § Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Earnings Quality & Accruals Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--limit", type=int, default=500, help="Universe size (default 500)")
    ap.add_argument("--min-mktcap", type=float, default=2e9, dest="min_mktcap",
                    help="Minimum market cap in dollars (default 2e9)")
    ap.add_argument("--open", action="store_true", help="Open the HTML report in a browser")
    args = ap.parse_args()

    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))

    print("\n" + "=" * 58)
    print("  EARNINGS QUALITY & ACCRUALS SCREENER")
    print("=" * 58)
    print(f"\nFetching universe (top {args.limit} US large caps)...")
    try:
        df = fetch_universe(args.limit, args.min_mktcap)
    except Exception as e:
        print(f"  FAILED to fetch universe: {e}")
        sys.exit(1)
    print(f"  {len(df)} stocks fetched.")

    rows = []
    for _, r in df.iterrows():
        sr = score_row(r)
        if sr is not None:
            rows.append(sr)
    rows.sort(key=lambda x: x["score"], reverse=True)
    print(f"  {len(rows)} stocks scored.\n")

    # Console: top 15 and bottom 10 (ASCII only)
    def _p(v, s="", dp=1):
        return f"{v:.{dp}f}{s}" if isinstance(v, (int, float)) else "--"
    print("  TOP 15 -- highest earnings quality")
    print("  " + "-" * 72)
    print("  {:<6} {:>5} {:<10} {:>9} {:>8} {:>8}  {}".format(
        "TICK", "SCORE", "TIER", "ACCRUAL%", "CFO/NI", "FCF/NI", "SECTOR"))
    for r in rows[:15]:
        print("  {:<6} {:>5.0f} {:<10} {:>9} {:>8} {:>8}  {}".format(
            r["ticker"], r["score"], r["tier"],
            _p(r["accruals"], "", 1), _p(r["cash_conv"], "", 2),
            _p(r["fcf_conv"], "", 2), r["sector"][:20]))

    print("\n  BOTTOM 10 -- weakest earnings quality / red flags")
    print("  " + "-" * 72)
    for r in rows[-10:][::-1]:
        flag = r["flags"][0] if r["flags"] else ""
        print("  {:<6} {:>5.0f} {:<10} {:>9}  {}".format(
            r["ticker"], r["score"], r["tier"],
            _p(r["accruals"], "", 1), flag[:38]))

    # Write HTML + CSV
    ts = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "earningsQualityData")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    html = build_html(rows, ts, len(df), suite_port=suite_port)
    fpath = os.path.join(out_dir, f"earnings_quality_{stamp}.html")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)

    csv_cols = ["ticker", "company", "sector", "price", "mktcap_bn", "score", "tier",
                "accruals", "cash_conv", "fcf_conv", "gross_margin", "net_margin", "roa", "de"]
    csv_path = os.path.join(out_dir, f"earnings_quality_{stamp}.csv")
    pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in rows]).to_csv(csv_path, index=False)

    kb = os.path.getsize(fpath) // 1024
    print(f"\n  HTML saved: {fpath}  ({kb} KB)")
    print(f"  CSV saved:  {csv_path}")
    if args.open:
        webbrowser.open("file://" + fpath)


if __name__ == "__main__":
    main()
