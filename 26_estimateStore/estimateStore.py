"""
Point-in-Time Analyst Estimate Store
====================================
Captures a timestamped snapshot of analyst estimates for a universe of US
large caps and appends it to a local SQLite database.  Over time this builds
a point-in-time (PIT) history so backtests can look up *what analysts actually
estimated as of a past date* -- instead of applying today's estimates to
historical dates, which is look-ahead bias.

Why it matters
--------------
The suite's backtests (valuationMaster, growthScreener) currently reuse today's
forward EPS / revenue / price targets for every historical snapshot, because no
point-in-time estimate source is available for free.  Estimates get revised
constantly, so this inflates the apparent accuracy of any method that leans on
them.  This tool is the fix: run it on a schedule (e.g. monthly) and the store
accumulates real PIT estimates that `get_estimate_asof()` can serve back.

What is captured (per ticker, per snapshot date)
------------------------------------------------
  price, market cap, forward EPS (next FY), forward revenue (next FY),
  TTM EPS, TTM revenue, TTM revenue growth, 1-yr price target, analyst rating.

USAGE
-----
  python estimateStore.py                     # capture today's snapshot (top 500)
  python estimateStore.py --capture --limit 800
  python estimateStore.py --query NVDA        # print stored history for a ticker
  python estimateStore.py --report --open     # HTML coverage + revision report
  python estimateStore.py --force             # re-capture even if today exists

Schedule it (monthly) via the suite's /schedule tooling or cron so the PIT
history grows.  Programmatic access:  from estimateStore import get_estimate_asof
"""

import sys, os, io, argparse, sqlite3, datetime, webbrowser
import pandas as pd

BG = "#0f1117"; S1 = "#161a27"; S2 = "#1a1e2e"; BD = "#252a3a"
TEXT = "#e2e8f0"; SUB = "#94a3b8"; GRN = "#22c55e"; RED = "#ef4444"; LINE1 = "#6366f1"

_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimateStoreData")
_DB    = os.path.join(_DIR, "estimates.db")

# TradingView estimate fields (all validated to return data)
FIELDS = [
    "name", "close", "market_cap_basic",
    "earnings_per_share_forecast_next_fy", "revenue_forecast_next_fy",
    "earnings_per_share_diluted_ttm", "total_revenue",
    "total_revenue_yoy_growth_ttm", "price_target_1y", "Recommend.All",
]

# DB column  <-  TradingView field
_COLMAP = [
    ("price",           "close"),
    ("market_cap",      "market_cap_basic"),
    ("fwd_eps",         "earnings_per_share_forecast_next_fy"),
    ("fwd_rev",         "revenue_forecast_next_fy"),
    ("eps_ttm",         "earnings_per_share_diluted_ttm"),
    ("rev_ttm",         "total_revenue"),
    ("rev_growth_pct",  "total_revenue_yoy_growth_ttm"),
    ("price_target",    "price_target_1y"),
    ("analyst_rating",  "Recommend.All"),
]


def _connect():
    os.makedirs(_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS estimates (
            snapshot_date  TEXT,
            ticker         TEXT,
            price          REAL,
            market_cap     REAL,
            fwd_eps        REAL,
            fwd_rev        REAL,
            eps_ttm        REAL,
            rev_ttm        REAL,
            rev_growth_pct REAL,
            price_target   REAL,
            analyst_rating REAL,
            captured_at    TEXT,
            PRIMARY KEY (snapshot_date, ticker)
        )
    """)
    conn.commit()
    return conn


# ── § Capture ─────────────────────────────────────────────────────────────────
def capture(limit=500, min_mktcap=2e9, force=False):
    from tradingview_screener import Query, col
    today = datetime.date.today().isoformat()

    conn = _connect()
    existing = conn.execute(
        "SELECT COUNT(*) FROM estimates WHERE snapshot_date=?", (today,)).fetchone()[0]
    if existing and not force:
        print(f"  Snapshot for {today} already exists ({existing} rows). "
              f"Use --force to overwrite.")
        conn.close()
        return existing

    print(f"  Fetching universe (top {limit} US large caps)...")
    q = (Query().select(*FIELDS).set_markets("america")
         .where(col("market_cap_basic") > min_mktcap, col("is_primary") == True)  # noqa: E712
         .order_by("market_cap_basic", ascending=False).limit(int(limit)))
    _, df = q.get_scanner_data()
    df = df.dropna(subset=["market_cap_basic"])
    print(f"  {len(df)} stocks fetched.")

    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        def g(f):
            v = r.get(f)
            try:
                v = float(v)
                return v if v == v else None
            except (TypeError, ValueError):
                return None
        rows.append((today, str(r.get("name", "?")),
                     *[g(tv) for _db, tv in _COLMAP], now))

    conn.executemany(
        "INSERT OR REPLACE INTO estimates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

    n_snaps = conn.execute("SELECT COUNT(DISTINCT snapshot_date) FROM estimates").fetchone()[0]
    dmin, dmax = conn.execute(
        "SELECT MIN(snapshot_date), MAX(snapshot_date) FROM estimates").fetchone()
    conn.close()
    print(f"  Stored {len(rows)} rows for {today}.")
    print(f"  Store now holds {n_snaps} snapshot date(s): {dmin} -> {dmax}")
    return len(rows)


# ── § Query / programmatic access ─────────────────────────────────────────────
def get_estimate_asof(ticker, as_of_date, conn=None):
    """
    Return the most recent estimate snapshot for `ticker` on or before
    `as_of_date` (a 'YYYY-MM-DD' string or date).  Returns a dict or None.

    This is the integration point for point-in-time backtests: instead of
    applying today's fwd_eps/fwd_rev to a historical date, call this to get the
    estimate that was actually on record at that time (if the store has it).
    """
    if hasattr(as_of_date, "isoformat"):
        as_of_date = as_of_date.isoformat()
    own = conn is None
    if own:
        conn = _connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM estimates WHERE ticker=? AND snapshot_date<=? "
        "ORDER BY snapshot_date DESC LIMIT 1",
        (ticker.upper(), as_of_date)).fetchone()
    if own:
        conn.close()
    return dict(row) if row else None


def query_ticker(ticker):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    hist = conn.execute(
        "SELECT * FROM estimates WHERE ticker=? ORDER BY snapshot_date",
        (ticker.upper(),)).fetchall()
    conn.close()
    if not hist:
        print(f"  No stored estimates for {ticker.upper()} yet.")
        return
    print(f"\n  Estimate history for {ticker.upper()}  ({len(hist)} snapshots)")
    print("  " + "-" * 74)
    print("  {:<12} {:>9} {:>9} {:>12} {:>9} {:>7}".format(
        "DATE", "FWD_EPS", "FWD_REV$B", "PRICE_TGT", "PRICE", "RATING"))
    prev = None
    for r in hist:
        fe  = r["fwd_eps"]; fr = r["fwd_rev"]; pt = r["price_target"]
        rev_b = (fr / 1e9) if fr else None
        rev_flag = ""
        if prev is not None and prev["fwd_eps"] and fe:
            d = (fe - prev["fwd_eps"]) / abs(prev["fwd_eps"]) * 100
            rev_flag = f"  ({d:+.1f}% EPS rev)" if abs(d) > 0.5 else ""
        print("  {:<12} {:>9} {:>9} {:>12} {:>9} {:>7}{}".format(
            r["snapshot_date"],
            f"{fe:.2f}" if fe else "--",
            f"{rev_b:.1f}" if rev_b else "--",
            f"{pt:.2f}" if pt else "--",
            f"{r['price']:.2f}" if r["price"] else "--",
            f"{r['analyst_rating']:+.2f}" if r["analyst_rating"] is not None else "--",
            rev_flag))
        prev = r


# ── § Report ──────────────────────────────────────────────────────────────────
def build_report(open_it=False):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    snaps = conn.execute(
        "SELECT snapshot_date, COUNT(*) n, MIN(captured_at) cap "
        "FROM estimates GROUP BY snapshot_date ORDER BY snapshot_date").fetchall()
    if not snaps:
        print("  Store is empty -- run a capture first.")
        conn.close()
        return None

    total = conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0]
    n_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM estimates").fetchone()[0]

    # Biggest EPS revisions between the two most recent snapshots (if >=2)
    movers = []
    if len(snaps) >= 2:
        d_new, d_old = snaps[-1]["snapshot_date"], snaps[-2]["snapshot_date"]
        cur = conn.execute("""
            SELECT a.ticker, a.fwd_eps new_eps, b.fwd_eps old_eps,
                   a.price_target new_pt, b.price_target old_pt
            FROM estimates a JOIN estimates b
              ON a.ticker=b.ticker
            WHERE a.snapshot_date=? AND b.snapshot_date=?
              AND a.fwd_eps IS NOT NULL AND b.fwd_eps IS NOT NULL AND b.fwd_eps<>0
        """, (d_new, d_old)).fetchall()
        for r in cur:
            pct = (r["new_eps"] - r["old_eps"]) / abs(r["old_eps"]) * 100
            movers.append((r["ticker"], pct, r["old_eps"], r["new_eps"]))
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
    conn.close()

    rows_html = "".join(
        f'<tr><td>{s["snapshot_date"]}</td><td class="r">{s["n"]}</td>'
        f'<td style="color:{SUB}">{(s["cap"] or "")[:19]}</td></tr>'
        for s in snaps)

    movers_html = ""
    if movers:
        mv = "".join(
            f'<tr><td class="tk">{t}</td>'
            f'<td class="r" style="color:{GRN if p>0 else RED}">{p:+.1f}%</td>'
            f'<td class="r">{oe:.2f}</td><td class="r">{ne:.2f}</td></tr>'
            for t, p, oe, ne in movers[:25])
        movers_html = f"""
<h2>Largest forward-EPS revisions (latest 2 snapshots)</h2>
<table><thead><tr><th>Ticker</th><th class="r">EPS revision</th>
<th class="r">Prior</th><th class="r">Current</th></tr></thead><tbody>{mv}</tbody></table>"""

    ts = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIT Estimate Store &mdash; {ts}</title><style>
:root{{--bg:{BG};--s1:{S1};--bd:{BD};--text:{TEXT};--sub:{SUB}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:{BG};color:{TEXT};padding:0 0 60px}}
.hero{{padding:26px 30px 20px;background:linear-gradient(135deg,{BG},{S1} 70%);border-bottom:1px solid {BD}}}
.hero h1{{font-size:21px;font-weight:800}}
.hero .sub{{font-size:12px;color:{SUB};margin-top:4px}}
.stats{{display:flex;gap:30px;margin-top:16px;flex-wrap:wrap}}
.sv{{font-size:22px;font-weight:800;color:{LINE1}}}.sl{{font-size:10px;color:{SUB};text-transform:uppercase;letter-spacing:1px}}
.wrap{{padding:22px 30px;max-width:900px}}
h2{{font-size:14px;margin:22px 0 10px;color:{TEXT}}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:10px}}
th,td{{padding:6px 10px;border-bottom:1px solid {BD};text-align:left}}
th{{color:{SUB};font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}}
td.r,th.r{{text-align:right}}td.tk{{font-weight:700}}
.note{{font-size:11px;color:{SUB};margin-top:8px;line-height:1.5}}
</style></head><body>
<div class="hero"><h1>Point-in-Time Estimate Store</h1>
<div class="sub">Generated {ts} &bull; Timestamped analyst estimates for PIT backtesting</div>
<div class="stats">
  <div><div class="sv">{len(snaps)}</div><div class="sl">Snapshot dates</div></div>
  <div><div class="sv">{total:,}</div><div class="sl">Total rows</div></div>
  <div><div class="sv">{n_tickers:,}</div><div class="sl">Unique tickers</div></div>
</div></div>
<div class="wrap">
<h2>Snapshot history</h2>
<table><thead><tr><th>Date</th><th class="r">Tickers</th><th>Captured at</th></tr></thead>
<tbody>{rows_html}</tbody></table>
{movers_html}
<div class="note">Programmatic access: <code>from estimateStore import get_estimate_asof</code> returns the
estimate on record on-or-before any date &mdash; wire this into the valuationMaster / growthScreener
backtests to remove look-ahead bias. Schedule a monthly capture so the history accumulates.</div>
</div></body></html>"""

    fpath = os.path.join(_DIR, f"estimate_store_report_{datetime.datetime.now():%Y%m%d_%H%M}.html")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved: {fpath}")
    if open_it:
        webbrowser.open("file://" + fpath)
    return fpath


# ── § Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Point-in-Time Analyst Estimate Store",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--capture", action="store_true", help="Capture today's snapshot (default action)")
    ap.add_argument("--query", metavar="TICKER", help="Print stored estimate history for a ticker")
    ap.add_argument("--report", action="store_true", help="Build an HTML coverage/revision report")
    ap.add_argument("--limit", type=int, default=500, help="Universe size for capture (default 500)")
    ap.add_argument("--min-mktcap", type=float, default=2e9, dest="min_mktcap")
    ap.add_argument("--force", action="store_true", help="Re-capture even if today already stored")
    ap.add_argument("--open", action="store_true", help="Open the HTML report in a browser")
    args = ap.parse_args()

    print("\n" + "=" * 58)
    print("  POINT-IN-TIME ANALYST ESTIMATE STORE")
    print("=" * 58)

    if args.query:
        query_ticker(args.query)
        return
    if args.report:
        build_report(open_it=args.open)
        return

    # default action is capture
    try:
        capture(args.limit, args.min_mktcap, force=args.force)
    except Exception as e:
        print(f"  Capture FAILED: {e}")
        sys.exit(1)
    build_report(open_it=args.open)


if __name__ == "__main__":
    main()
