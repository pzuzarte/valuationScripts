#!/usr/bin/env python3
"""
watchlistTracker.py — Watchlist & Valuation History Tracker
=============================================================
Manages a persistent watchlist and valuation snapshot history in a local
SQLite database at ~/.valuation_suite/data.db.

This database is populated automatically:
  • Every valuationMaster.py run saves a valuation snapshot (regardless of watchlist)
  • Every portfolioAnalyzer.py run saves portfolio signals

Usage
-----
    python watchlistTracker.py                           # HTML report
    python watchlistTracker.py --add NVDA                # add to watchlist
    python watchlistTracker.py --add NVDA --target 150 --notes "AI play"
    python watchlistTracker.py --remove NVDA             # remove from watchlist
    python watchlistTracker.py --history NVDA            # print history to terminal
    python watchlistTracker.py --list                    # print watchlist to terminal
"""

import argparse
import datetime
import os
import sqlite3
import sys
import webbrowser

# ── Database ──────────────────────────────────────────────────────────────────

DB_DIR  = os.path.join(os.path.expanduser("~"), ".valuation_suite")
DB_PATH = os.path.join(DB_DIR, "data.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker       TEXT PRIMARY KEY,
    company_name TEXT,
    added_date   TEXT,
    target_price REAL,
    notes        TEXT
);
CREATE TABLE IF NOT EXISTS valuation_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    snapshot_date TEXT    NOT NULL,
    price         REAL,
    fair_value    REAL,
    upside_pct    REAL,
    verdict       TEXT,
    wacc          REAL,
    growth_rate   REAL,
    top_method    TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date  TEXT    NOT NULL,
    portfolio_file TEXT,
    ticker         TEXT    NOT NULL,
    signal         TEXT,
    fair_value     REAL,
    upside_pct     REAL
);
"""


def _conn():
    """Open (and initialise) the database connection."""
    os.makedirs(DB_DIR, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(_SCHEMA)
    c.commit()
    return c


# ── yfinance (optional) ───────────────────────────────────────────────────────

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False


def _fetch_price_name(ticker: str):
    """Return (price, company_name) via yfinance, or (None, None) on failure."""
    if not _YF:
        return None, None
    try:
        t     = yf.Ticker(ticker)
        fi    = t.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
        name  = (t.info or {}).get("shortName") or ticker
        return (float(price) if price else None), name
    except Exception:
        return None, None


# ── CLI operations ─────────────────────────────────────────────────────────────

def cmd_add(ticker: str, target: float = None, notes: str = None):
    ticker = ticker.upper().strip()
    price, name = _fetch_price_name(ticker)
    c = _conn()
    try:
        c.execute(
            """INSERT INTO watchlist (ticker, company_name, added_date, target_price, notes)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                   target_price = COALESCE(excluded.target_price, target_price),
                   notes        = COALESCE(excluded.notes, notes),
                   company_name = COALESCE(excluded.company_name, company_name)""",
            (ticker, name, datetime.date.today().isoformat(), target, notes),
        )
        c.commit()
        price_s = "${:.2f}".format(price) if price else "price unavailable"
        print("  Added {} ({}) at {}".format(ticker, name or "—", price_s))
        if target:
            print("  Target: ${:.2f}".format(target))
        if notes:
            print("  Notes: {}".format(notes))
    finally:
        c.close()


def cmd_remove(ticker: str):
    ticker = ticker.upper().strip()
    c = _conn()
    try:
        c.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        c.commit()
        print("  {} removed from watchlist".format(ticker))
    finally:
        c.close()


def cmd_list():
    c = _conn()
    try:
        rows = c.execute(
            """SELECT w.ticker, w.company_name, w.added_date, w.target_price, w.notes,
                      s.snapshot_date, s.fair_value, s.upside_pct, s.verdict
               FROM watchlist w
               LEFT JOIN (
                   SELECT ticker, snapshot_date, fair_value, upside_pct, verdict
                   FROM valuation_snapshots
                   WHERE id IN (SELECT MAX(id) FROM valuation_snapshots GROUP BY ticker)
               ) s ON s.ticker = w.ticker
               ORDER BY w.ticker"""
        ).fetchall()
    finally:
        c.close()

    if not rows:
        print("  Watchlist is empty. Use --add TICKER to add stocks.")
        return

    print("\n  {:8} {:28} {:12} {:>8} {:>10} {:>7}  {}".format(
        "Ticker", "Company", "Added", "Target", "Fair Val", "Upside", "Verdict"))
    print("  " + "─" * 80)
    for ticker, name, added, target, notes, snap_d, fv, upside, verdict in rows:
        t_s  = "${:.0f}".format(target)  if target else "—"
        fv_s = "${:.0f}".format(fv)      if fv     else "—"
        u_s  = "{:+.1f}%".format(upside) if upside else "—"
        v_s  = (verdict or "—")[:8]
        print("  {:8} {:28} {:12} {:>8} {:>10} {:>7}  {}".format(
            ticker, (name or "—")[:26], added or "—", t_s, fv_s, u_s, v_s))


def cmd_history(ticker: str):
    ticker = ticker.upper().strip()
    c = _conn()
    try:
        rows = c.execute(
            """SELECT snapshot_date, price, fair_value, upside_pct, verdict,
                      wacc, growth_rate, top_method
               FROM valuation_snapshots
               WHERE ticker = ?
               ORDER BY snapshot_date DESC
               LIMIT 25""",
            (ticker,),
        ).fetchall()
    finally:
        c.close()

    if not rows:
        print("  No history for {}. Run: python valuationMaster.py {}".format(ticker, ticker))
        return

    print("\n  Valuation history — {} ({} snapshots)\n".format(ticker, len(rows)))
    print("  {:12} {:>8} {:>10} {:>8} {:12} {:>6} {:>7}  Method".format(
        "Date", "Price", "Fair Val", "Upside", "Verdict", "WACC", "Growth"))
    print("  " + "─" * 80)
    for snap_d, price, fv, upside, verdict, wacc, growth, method in rows:
        print("  {:12} {:>8} {:>10} {:>8} {:12} {:>6} {:>7}  {}".format(
            snap_d,
            "${:.2f}".format(price)         if price  else "—",
            "${:.2f}".format(fv)            if fv     else "—",
            "{:+.1f}%".format(upside)       if upside else "—",
            (verdict or "—")[:12],
            "{:.1f}%".format(wacc * 100)    if wacc   else "—",
            "{:.1f}%".format(growth * 100)  if growth else "—",
            (method or "—")[:20],
        ))


# ── HTML report helpers ────────────────────────────────────────────────────────

def _mo(v):
    return "${:.2f}".format(v) if v is not None else "—"

def _pc(v):
    return "{:+.1f}%".format(v) if v is not None else "—"


def _sparkline(history: list) -> str:
    """Tiny SVG sparkline from [(date, fv), ...] sorted ascending."""
    vals = [fv for _, fv in history if fv is not None]
    if len(vals) < 2:
        return "<span style='color:#666;font-size:10px;font-family:monospace;'>no history</span>"
    W, H   = 80, 22
    mn, mx = min(vals), max(vals)
    rng    = mx - mn or 1
    pts    = []
    for i, v in enumerate(vals):
        x = int(i / (len(vals) - 1) * W)
        y = int((1 - (v - mn) / rng) * (H - 4)) + 2
        pts.append("{},{}".format(x, y))
    color = "#4aac6e" if vals[-1] >= vals[0] else "#e05252"
    return (
        "<svg width='{W}' height='{H}' style='display:inline-block;vertical-align:middle;'>"
        "<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='1.5'/>"
        "</svg>"
    ).format(W=W, H=H, pts=" ".join(pts), c=color)


def _history_cards(latest_val: dict, history_fv: dict) -> str:
    """Build one card per ticker that has 2+ snapshots."""
    tickers = sorted(
        [t for t in history_fv if len(history_fv[t]) >= 2],
        key=lambda t: -len(history_fv[t]),
    )
    if not tickers:
        return (
            "<p style='color:var(--muted);font-size:14px;'>"
            "No historical snapshots yet. "
            "Run <code>python valuationMaster.py TICKER</code> to create snapshots."
            "</p>"
        )

    cards = ""
    for ticker in tickers[:30]:
        hist   = sorted(history_fv[ticker], key=lambda x: x[0])
        latest = latest_val.get(ticker, {})
        spark  = _sparkline(hist)

        snap_rows = "".join(
            "<tr>"
            "<td style='padding:3px 10px;font-size:11px;color:#888;'>{d}</td>"
            "<td style='padding:3px 10px;font-family:DM Mono,monospace;font-size:12px;'>{fv}</td>"
            "</tr>".format(d=d, fv=_mo(fv))
            for d, fv in reversed(hist[-12:])
        )

        cards += (
            "<div style='background:var(--surface);border-radius:10px;"
            "padding:20px 24px;margin-bottom:12px;display:inline-block;"
            "min-width:260px;vertical-align:top;margin-right:12px;'>"
            "<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
            "<span style='font-size:16px;font-weight:700;'>{tkr}</span>"
            "{spark}"
            "<span style='font-size:11px;color:var(--muted);'>{n} snapshots &nbsp; latest: {fv}</span>"
            "</div>"
            "<table style='border-collapse:collapse;'>"
            "<thead><tr>"
            "<th style='padding:3px 10px;font-size:9px;letter-spacing:1.5px;"
            "text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);'>"
            "Date</th>"
            "<th style='padding:3px 10px;font-size:9px;letter-spacing:1.5px;"
            "text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);'>"
            "Fair Value</th>"
            "</tr></thead>"
            "<tbody>{rows}</tbody>"
            "</table>"
            "</div>"
        ).format(
            tkr=ticker, spark=spark,
            n=len(hist), fv=_mo(latest.get("fv")),
            rows=snap_rows,
        )
    return "<div style='overflow-x:auto;'>" + cards + "</div>"


def generate_report() -> str:
    """Build the watchlist HTML report string."""
    c = _conn()
    try:
        watched = c.execute(
            "SELECT ticker, company_name, added_date, target_price, notes "
            "FROM watchlist ORDER BY ticker"
        ).fetchall()

        # Latest valuation snapshot per ticker (all tickers ever analysed)
        latest_val = {}
        for row in c.execute(
            "SELECT ticker, snapshot_date, price, fair_value, upside_pct, "
            "verdict, wacc, growth_rate, top_method "
            "FROM valuation_snapshots "
            "WHERE id IN (SELECT MAX(id) FROM valuation_snapshots GROUP BY ticker)"
        ).fetchall():
            latest_val[row[0]] = {
                "date": row[1], "price": row[2], "fv": row[3], "upside": row[4],
                "verdict": row[5], "wacc": row[6], "growth": row[7], "method": row[8],
            }

        # Latest portfolio signal per ticker
        latest_sig = {
            ticker: sig
            for ticker, sig in c.execute(
                "SELECT ticker, signal FROM portfolio_signals "
                "WHERE id IN (SELECT MAX(id) FROM portfolio_signals GROUP BY ticker)"
            ).fetchall()
        }

        # Historical fair values per ticker
        history_fv = {}
        for ticker, date, fv in c.execute(
            "SELECT ticker, snapshot_date, fair_value FROM valuation_snapshots "
            "WHERE fair_value IS NOT NULL ORDER BY ticker, snapshot_date"
        ).fetchall():
            history_fv.setdefault(ticker, []).append((date, fv))

    finally:
        c.close()

    # Fetch live prices for watchlist tickers
    live_prices = {}
    if _YF and watched:
        for ticker, *_ in watched:
            p, _ = _fetch_price_name(ticker)
            if p:
                live_prices[ticker] = p

    # Summary stats
    n_total = len(watched)
    n_buy   = sum(
        1 for t, *_ in watched
        if (latest_sig.get(t, "") or "").upper().startswith("BUY")
        or (latest_val.get(t, {}).get("verdict") or "").upper().startswith("BUY")
    )
    today_str = datetime.date.today().isoformat()

    # Build table rows
    rows_html = ""
    for ticker, company, added_date, target, notes in watched:
        live    = live_prices.get(ticker)
        val     = latest_val.get(ticker, {})
        sig     = latest_sig.get(ticker, "")
        fv      = val.get("fv")
        upside  = val.get("upside")
        verdict = val.get("verdict", "")
        snap_d  = val.get("date")

        if snap_d:
            try:
                delta   = (datetime.date.today() - datetime.date.fromisoformat(snap_d)).days
                age_s   = "today" if delta == 0 else "{} days ago".format(delta)
                age_col = "var(--up)" if delta <= 7 else "var(--warn)" if delta <= 30 else "var(--muted)"
            except Exception:
                age_s, age_col = "—", "var(--muted)"
        else:
            age_s, age_col = "No snapshot", "var(--muted)"

        signal_d = ((sig or verdict) or "—").upper()[:6]
        _sig_colors = {
            "BUY":  ("var(--up)",   "#0f1117"),
            "HOLD": ("var(--warn)", "#0f1117"),
            "SELL": ("var(--down)", "#fff"),
        }
        sig_bg, sig_fg = _sig_colors.get(signal_d[:4], ("#444", "#fff"))

        mos_fv  = fv * 0.80 if fv else None
        u_col   = "var(--up)" if (upside and upside > 0) else "var(--down)" if upside else "var(--muted)"
        row_bdr = ""
        if signal_d.startswith("BUY"):
            row_bdr = "border-left:3px solid var(--up);"
        elif signal_d.startswith("SEL"):
            row_bdr = "border-left:3px solid var(--down);"
        elif signal_d.startswith("HOL"):
            row_bdr = "border-left:3px solid var(--warn);"

        spark = _sparkline(history_fv.get(ticker, []))

        rows_html += (
            "<tr style='{rb}'>"
            "<td style='padding:11px 16px;font-weight:700;'>{tkr}</td>"
            "<td style='padding:11px 16px;font-size:12px;color:var(--muted);'>{co}</td>"
            "<td style='padding:11px 16px;font-family:DM Mono,monospace;'>{lp}</td>"
            "<td style='padding:11px 16px;font-family:DM Mono,monospace;'>{fv_s}</td>"
            "<td style='padding:11px 16px;font-family:DM Mono,monospace;color:{uc};'>{up_s}</td>"
            "<td style='padding:11px 16px;font-family:DM Mono,monospace;'>{tg}</td>"
            "<td style='padding:11px 16px;font-family:DM Mono,monospace;'>{mos}</td>"
            "<td style='padding:11px 16px;'>"
            "<span style='background:{sb};color:{sf};padding:3px 10px;border-radius:4px;"
            "font-size:11px;font-weight:700;letter-spacing:1px;'>{sd}</span>"
            "</td>"
            "<td style='padding:11px 16px;'>{spk}</td>"
            "<td style='padding:11px 16px;font-size:11px;color:{ac};'>{age}</td>"
            "<td style='padding:11px 16px;font-size:11px;color:var(--muted);'>{nt}</td>"
            "</tr>"
        ).format(
            rb=row_bdr, tkr=ticker,
            co=(company or "—")[:28],
            lp=_mo(live), fv_s=_mo(fv),
            uc=u_col, up_s=_pc(upside),
            tg=_mo(target) if target else "—",
            mos=_mo(mos_fv),
            sb=sig_bg, sf=sig_fg, sd=signal_d,
            spk=spark,
            ac=age_col, age=age_s,
            nt=(notes or "")[:40],
        )

    if not rows_html:
        rows_html = (
            "<tr><td colspan='11' style='padding:32px;text-align:center;color:var(--muted);'>"
            "No tickers on watchlist. Run: "
            "<code>python watchlistTracker.py --add TICKER</code>"
            "</td></tr>"
        )

    hist_cards = _history_cards(latest_val, history_fv)
    ts = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    return _REPORT_HTML.format(
        ts=ts, today=today_str,
        n_total=n_total, n_buy=n_buy,
        rows=rows_html,
        history_section=hist_cards,
    )


# ── HTML template ──────────────────────────────────────────────────────────────

_REPORT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Watchlist Report &mdash; {today}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:      #0f1117; --surface: #16192b; --surface2: #1c2038;
  --border:  #252a45; --text:    #e8eaf6; --muted:    #7b8299;
  --accent:  #4f8ef7; --up:      #4aac6e; --down:     #e05252; --warn:    #f5a623;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: Syne, sans-serif; min-height: 100vh; }}
.header {{ background: linear-gradient(135deg, #0d1526, #1a1f3a); padding: 40px 48px; border-bottom: 1px solid var(--border); }}
.header h1 {{ font-family: "DM Serif Display", serif; font-size: 28px; }}
.header .meta {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
.container {{ max-width: 1440px; margin: 0 auto; padding: 32px 48px; }}
.section-label {{ font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
                  color: var(--muted); margin: 32px 0 14px;
                  padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
.stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 8px; }}
.stat-card {{ background: var(--surface); border-radius: 10px; padding: 20px 24px; }}
.stat-card .lbl {{ font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
.stat-card .val {{ font-size: 28px; font-family: "DM Mono", monospace; font-weight: 700; }}
.tbl-wrap {{ overflow-x: auto; background: var(--surface); border-radius: 10px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ padding: 10px 16px; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
      color: var(--muted); border-bottom: 2px solid var(--border); text-align: left; white-space: nowrap; }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
.footer {{ text-align: center; padding: 32px; font-size: 11px; color: var(--muted);
           border-top: 1px solid var(--border); margin-top: 40px; }}
code {{ background: var(--surface2); padding: 1px 5px; border-radius: 3px; font-family: "DM Mono", monospace; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Watchlist &amp; Valuation History</h1>
  <div class="meta">Generated {ts} &nbsp;&middot;&nbsp; DB: ~/.valuation_suite/data.db</div>
</div>

<div class="container">

  <div class="stat-grid">
    <div class="stat-card">
      <div class="lbl">Watched Tickers</div>
      <div class="val" style="color:var(--accent);">{n_total}</div>
    </div>
    <div class="stat-card">
      <div class="lbl">BUY Signal Today</div>
      <div class="val" style="color:var(--up);">{n_buy}</div>
    </div>
    <div class="stat-card">
      <div class="lbl">Report Date</div>
      <div class="val" style="font-size:18px;color:var(--text);">{today}</div>
    </div>
  </div>

  <div class="section-label">Watchlist &mdash; All Tracked Holdings</div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Company</th>
          <th>Live Price</th>
          <th>Fair Value</th>
          <th>Upside</th>
          <th>Target Price</th>
          <th>20% MoS</th>
          <th>Signal</th>
          <th>FV Trend</th>
          <th>Snapshot Age</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="section-label">Valuation History &mdash; All Analysed Tickers</div>
  {history_section}

</div>

<div class="footer">
  Watchlist Tracker &middot; ValuationSuite &middot; {ts}<br>
  Live prices: yfinance &middot; History: SQLite at ~/.valuation_suite/data.db<br>
  <span style="opacity:0.5;">Informational only. Not financial advice.</span>
</div>

</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watchlist & Valuation History Tracker",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--add",     metavar="TICKER", help="Add ticker to watchlist")
    parser.add_argument("--remove",  metavar="TICKER", help="Remove ticker from watchlist")
    parser.add_argument("--history", metavar="TICKER", help="Print valuation history for ticker")
    parser.add_argument("--list",    action="store_true", help="Print watchlist to terminal")
    parser.add_argument("--target",  type=float, metavar="PRICE",
                        help="Target price (used with --add)")
    parser.add_argument("--notes",   metavar="TEXT",
                        help="Notes (used with --add)")

    args = parser.parse_args()

    print("\n" + "=" * 56)
    print("  WATCHLIST TRACKER")
    print("=" * 56)
    print("  DB: {}".format(DB_PATH))

    if args.add:
        cmd_add(args.add, target=args.target, notes=args.notes)

    elif args.remove:
        cmd_remove(args.remove)

    elif args.history:
        cmd_history(args.history)

    elif args.list:
        cmd_list()

    else:
        # No subcommand → generate HTML report
        print("\nGenerating watchlist report...")
        html = generate_report()

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlistData")
        os.makedirs(outdir, exist_ok=True)
        date_prefix = datetime.datetime.now().strftime("%Y_%m_%d")
        outfile = os.path.join(outdir, date_prefix + "_watchlist.html")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(html)

        print("  Saved: {}".format(outfile))
        if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
            print("\nOpening report in browser...")
            webbrowser.open("file://" + os.path.abspath(outfile))

    print("\nDone.")


if __name__ == "__main__":
    main()
