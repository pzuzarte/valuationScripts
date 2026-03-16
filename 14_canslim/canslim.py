"""
CANSLIM Screener — O'Neil's Growth Stock Methodology
=====================================================
Scores stocks across the six CANSLIM factors using TradingView data:

  C  Current quarterly EPS growth   — EPS YoY growth TTM ≥25%       (30 pts)
  A  Annual earnings trend          — ROE as quality/consistency proxy (20 pts)
  N  Near 52-week high              — (close / 52w_high) ≥75%         (20 pts)
  S  Supply & demand (size)         — Growth-stage market cap $1B-$15B (15 pts)
  L  Leader vs laggard              — 3-month relative strength        (25 pts)
  I  Institutional sponsorship      — Approx: not over-owned/neglected (10 pts)

Total: 100 pts.  Minimum to appear: N ≥55%, 3M perf ≥0%, EPS growth ≥0%.

Grades:  A+(≥88) · A(≥75) · B+(≥60) · B(≥45) · C(<45)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMAND LINE ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --index  INDEX    SPX | NDX | RUT | TSX   (default: SPX)
  --top    N        Limit fetch to top N by market cap
  --min-score N     Only show stocks with CANSLIM score ≥ N  (default: 0)
  --csv             Export results to CSV
  --help            Print this help and exit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python canslim.py --index NDX
  python canslim.py --index SPX --min-score 60
  python canslim.py --index RUT --top 500 --csv
"""

import sys, math, datetime, webbrowser, os, csv, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tradingview_screener import Query, col
except ImportError:
    print("ERROR: tradingview-screener not installed.\nRun:  pip install tradingview-screener")
    sys.exit(1)

# ── FIELDS ─────────────────────────────────────────────────────────────────────
FIELDS_CORE = [
    "name", "close", "market_cap_basic", "sector",
    "earnings_per_share_diluted_yoy_growth_ttm",   # C factor
    "total_revenue_yoy_growth_ttm",                # revenue growth
    "return_on_equity",                            # A factor proxy
    "gross_margin", "operating_margin",            # quality
    "price_52_week_high", "price_52_week_low",     # N factor
    "Perf.1M", "Perf.3M",                          # L factor
    "price_earnings_ttm",
    "beta_1_year",
]

FIELDS_V3 = [
    "return_on_invested_capital",
    "earnings_per_share_forecast_next_fy",
]

EXCLUDE_SECTORS = {"Financials", "Finance", "Utilities"}
MIN_MARKET_CAP  = 200e6   # $200M minimum

# ── INDEX CONFIG ───────────────────────────────────────────────────────────────
NASDAQ100 = [
    "ADBE","AMD","ABNB","GOOGL","GOOG","AMZN","AEP","AMGN","ADI","ANSS","AAPL","AMAT","APP",
    "ASML","AZN","TEAM","ADSK","ADP","AXON","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CTAS",
    "CSCO","CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX","DXCM","FANG","DDOG",
    "DLTR","EA","EXC","FAST","FTNT","GEHC","GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG",
    "KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI","META","MCHP","MU","MSFT","MRNA",
    "MDLZ","MDB","MNST","NFLX","NVDA","NXPI","ORLY","ON","PCAR","PANW","PAYX","PYPL","PDD","QCOM",
    "REGN","ROP","ROST","SBUX","SNPS","TTWO","TMUS","TSLA","TXN","TTD","VRSK","VRTX","WBD","WBA",
    "WDAY","XEL","ZS","ZM",
]

TSX = [
    "SHOP","RY","TD","ENB","CP","CNR","BN","BAM","BCE","BMO","BNS","MFC","SLF","TRI","ABX",
    "CCO","IMO","CNQ","SU","CVE","PPL","TRP","K","POW","GWO","IAG","FFH","L","EMA","FTS",
    "H","CAR","REI","CHP","HR","SRU","AP","DIR","NWH","CSH","MRG","CRT","WPM","AEM","AGI",
    "KL","FNV","OR","EDV","MAG","ELD","SSL","IMG","PVG","OGC","TMX","X","ACO","MG","MDA",
    "OTEX","DSG","GIB","BB","CIGI","PHO","ATD","DOL","CTC","MRU","EMP","SAP","QSR","MTY",
    "TFII","TIH","WSP","STN","ATA","BYD","NFI","TDG","RBA","LIF","CCL","ITP","TCL","TXF",
    "WCN","BIN","GFL","SNC","STLC","HPS","FTT","IFP","PRE","FSV","TIXT","AC","CAE","CCA",
]

INDEX_CONFIG = {
    "SPX": {"name": "S&P 500",      "tickers": None,     "cap_limit": 503,  "is_rut": False, "is_tsx": False},
    "NDX": {"name": "Nasdaq 100",   "tickers": NASDAQ100,"cap_limit": 110,  "is_rut": False, "is_tsx": False},
    "RUT": {"name": "Russell 2000", "tickers": None,     "cap_limit": 2000, "is_rut": True,  "is_tsx": False},
    "TSX": {"name": "TSX",          "tickers": TSX,      "cap_limit": 200,  "is_rut": False, "is_tsx": True},
}

# ── FETCH ──────────────────────────────────────────────────────────────────────
def fetch_stocks(index_code, limit=None):
    import pandas as pd
    cfg     = INDEX_CONFIG.get(index_code.upper(), INDEX_CONFIG["SPX"])
    name    = cfg["name"]
    tickers = cfg.get("tickers")
    cap     = limit or cfg["cap_limit"]
    tv_market = "canada" if cfg["is_tsx"] else None
    print(f"  Querying TradingView for {name}...")

    def _q():
        q = Query()
        if tv_market:
            q = q.set_markets(tv_market)
        return q

    def _merge_v3(df):
        try:
            ticks = df["name"].tolist()
            _, v3df = (_q().select("name", *FIELDS_V3)
                       .where(col("name").isin(ticks))
                       .limit(len(ticks) + 20).get_scanner_data())
            df = df.merge(v3df[["name"] + FIELDS_V3], on="name", how="left")
            print(f"  v3 fields merged OK — {len(v3df)} rows")
        except Exception as e:
            print(f"  v3 fields unavailable ({str(e)[:80]}), continuing without")
        return df

    if tickers:
        try:
            lim = limit or len(tickers) + 20
            _, df = (_q().select(*FIELDS_CORE)
                     .where(col("name").isin(tickers), col("is_primary") == True)
                     .order_by("market_cap_basic", ascending=False)
                     .limit(lim).get_scanner_data())
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df = _merge_v3(df)
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks fetched"); return df
        except Exception as e:
            print(f"  isin filter failed ({str(e)[:60]}), trying fallback...")

    if cfg["is_tsx"]:
        try:
            _, df = (_q().select(*FIELDS_CORE)
                     .where(col("is_primary") == True,
                            col("typespecs").has_none_of(["preferred"]),
                            col("market_cap_basic") > MIN_MARKET_CAP)
                     .order_by("market_cap_basic", ascending=False)
                     .limit(cap + 50).get_scanner_data())
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df = _merge_v3(df)
            df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
            df["index_label"] = name
            print(f"  {name}: {len(df)} stocks (canada fallback)"); return df
        except Exception as e:
            print(f"  TSX fallback failed: {str(e)[:60]}")

    frames = []
    for exch in ["NYSE", "NASDAQ"]:
        try:
            _, df = (Query().select(*FIELDS_CORE)
                     .where(col("exchange") == exch, col("is_primary") == True,
                            col("typespecs").has_none_of(["preferred", "depositary"]),
                            col("market_cap_basic") > MIN_MARKET_CAP)
                     .order_by("market_cap_basic", ascending=False)
                     .limit(1500).get_scanner_data())
            frames.append(df); print(f"    {exch}: {len(df)} stocks")
        except Exception as e:
            print(f"    {exch} failed: {str(e)[:60]}")
    if not frames:
        return pd.DataFrame(columns=FIELDS_CORE)
    combined = pd.concat([f.reset_index(drop=True) for f in frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["name"], keep="first")
    combined = combined.sort_values("market_cap_basic", ascending=False).head(cap + 50).reset_index(drop=True)
    combined = _merge_v3(combined)
    combined = combined.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    combined["index_label"] = name
    print(f"  {name}: {len(combined)} stocks (NYSE+NASDAQ)"); return combined

# ── PARSE ──────────────────────────────────────────────────────────────────────
def parse_row(row: dict) -> dict:
    def s(k, default=None):
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        try: return float(v)
        except (TypeError, ValueError): return default

    ticker  = str(row.get("name", "")).strip()
    price   = s("close")
    mktcap  = s("market_cap_basic")
    sector  = str(row.get("sector") or "")
    eps_g   = s("earnings_per_share_diluted_yoy_growth_ttm")  # % already? check below
    rev_g   = s("total_revenue_yoy_growth_ttm")
    roe     = s("return_on_equity")
    gross_m = s("gross_margin")
    op_m    = s("operating_margin")
    h52     = s("price_52_week_high")
    l52     = s("price_52_week_low")
    p1m     = s("Perf.1M")
    p3m     = s("Perf.3M")
    pe      = s("price_earnings_ttm")
    beta    = s("beta_1_year")
    roic    = s("return_on_invested_capital")
    fwd_eps = s("earnings_per_share_forecast_next_fy")

    if not ticker or not price or price <= 0:
        return {"ticker": None}

    # TradingView returns all growth/margin/return/performance fields as percentages already
    # N factor: position in 52-week range as % of high
    n_ratio = None
    if h52 and h52 > 0 and price:
        n_ratio = round(price / h52 * 100, 1)   # % of 52w high

    return {
        "ticker":    ticker,
        "price":     round(price, 2),
        "mktcap":    mktcap,
        "sector":    sector,
        "eps_g":     round(eps_g,   1) if eps_g   is not None else None,
        "rev_g":     round(rev_g,   1) if rev_g   is not None else None,
        "roe":       round(roe,     1) if roe     is not None else None,
        "roic":      round(roic,    1) if roic    is not None else None,
        "gross_m":   round(gross_m, 1) if gross_m is not None else None,
        "op_m":      round(op_m,    1) if op_m    is not None else None,
        "n_ratio":   n_ratio,
        "h52":       round(h52,  2)    if h52     is not None else None,
        "l52":       round(l52,  2)    if l52     is not None else None,
        "p1m":       round(p1m,  1)    if p1m     is not None else None,
        "p3m":       round(p3m,  1)    if p3m     is not None else None,
        "pe":        round(pe, 1)         if pe        is not None else None,
        "beta":      round(beta, 2)       if beta      is not None else None,
        "fwd_eps":   fwd_eps,
    }

# ── SCORE ──────────────────────────────────────────────────────────────────────
def score_stock(d: dict) -> dict | None:
    """Score a stock on CANSLIM criteria. Returns None if minimum criteria not met."""
    eps_g  = d.get("eps_g")
    p3m    = d.get("p3m")
    n_ratio = d.get("n_ratio")

    # Minimum gate — must pass all three to be displayed
    if eps_g  is None or eps_g  < 0:   return None
    if p3m    is None or p3m    < 0:   return None
    if n_ratio is None or n_ratio < 55: return None

    score = 0.0

    # ── C + A  (combined 30 pts: EPS growth rate) ─────────────────────────────
    c_pts = 0
    if   eps_g >= 50: c_pts = 30
    elif eps_g >= 25: c_pts = 24
    elif eps_g >= 15: c_pts = 16
    elif eps_g >= 5:  c_pts = 8
    score += c_pts

    # ── A  (20 pts: ROE as annual quality / consistency proxy) ────────────────
    roe = d.get("roe")
    a_pts = 0
    if roe is not None:
        if   roe >= 30: a_pts = 20
        elif roe >= 20: a_pts = 15
        elif roe >= 12: a_pts = 9
        elif roe >= 5:  a_pts = 4
    score += a_pts

    # ── N  (20 pts: near 52-week high) ────────────────────────────────────────
    n_pts = 0
    if   n_ratio >= 95: n_pts = 20
    elif n_ratio >= 88: n_pts = 17
    elif n_ratio >= 80: n_pts = 13
    elif n_ratio >= 70: n_pts = 8
    elif n_ratio >= 55: n_pts = 3
    score += n_pts

    # ── S  (15 pts: size — growth-stage sweet spot) ───────────────────────────
    mc = d.get("mktcap") or 0
    s_pts = 0
    if   mc >= 1e9 and mc < 15e9:  s_pts = 15   # $1B-$15B: O'Neil sweet spot
    elif mc >= 15e9 and mc < 50e9: s_pts = 10   # $15B-$50B: large but still moveable
    elif mc >= 200e6 and mc < 1e9: s_pts = 8    # <$1B: micro/small, higher risk
    else:                          s_pts = 4    # mega-cap (>$50B): harder to double
    score += s_pts

    # ── L  (25 pts: leader vs laggard — 3M RS + 1M continuation) ─────────────
    p1m = d.get("p1m")
    l_pts = 0
    if   p3m >= 30: l_pts = 25
    elif p3m >= 20: l_pts = 20
    elif p3m >= 10: l_pts = 14
    elif p3m >= 5:  l_pts = 8
    elif p3m >= 0:  l_pts = 3
    # Continuation bonus: 1M momentum still positive
    if p1m is not None and p1m >= 5:
        l_pts = min(25, l_pts + 4)
    score += l_pts

    # ── I  (10 pts: institutional quality proxy via ROIC / margin) ────────────
    roic = d.get("roic")
    i_pts = 0
    if roic is not None:
        if   roic >= 25: i_pts = 10
        elif roic >= 15: i_pts = 7
        elif roic >= 8:  i_pts = 4
        else:            i_pts = 1
    elif d.get("gross_m") is not None:
        gm = d["gross_m"]
        if   gm >= 60: i_pts = 8
        elif gm >= 40: i_pts = 5
        elif gm >= 20: i_pts = 2
    score += i_pts

    score = round(min(100.0, score), 1)

    if   score >= 88: grade = "A+"
    elif score >= 75: grade = "A"
    elif score >= 60: grade = "B+"
    elif score >= 45: grade = "B"
    else:             grade = "C"

    return {**d, "score": score, "grade": grade,
            "pts_c": c_pts, "pts_a": a_pts, "pts_n": n_pts,
            "pts_s": s_pts, "pts_l": l_pts, "pts_i": i_pts}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def _fmtpct(v, signed=False, dec=1):
    if v is None: return "—"
    s = f"{v:+.{dec}f}%" if signed else f"{v:.{dec}f}%"
    return s

def _fmtnum(v, dec=0):
    if v is None: return "—"
    return f"{v:.{dec}f}"

def _fmtmc(v):
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:.0f}"

def _grade_color(g):
    return {"A+": "#00c896", "A": "#4caf7a", "B+": "#cddc39",
            "B":  "#f0a500", "C": "#e05c5c"}.get(g, "#6b7194")

def _pct_color(v, lo=0, hi=25, invert=False):
    if v is None: return "#6b7194"
    good = v > hi if not invert else v < lo
    mid  = lo <= v <= hi if not invert else lo <= -v <= hi
    if good: return "#00c896"
    if mid:  return "#f0a500"
    return "#e05c5c"

def _score_color(v):
    if v is None: return "#6b7194"
    if v >= 75:   return "#00c896"
    if v >= 50:   return "#f0a500"
    return "#e05c5c"

# ── DEEP-DIVE WATCHLIST BAR ────────────────────────────────────────────────────
_WL_CSS = """
#wl-bar{display:flex;align-items:center;gap:10px;padding:7px 16px;
  background:#1a1e2e;border-top:1px solid #252a3a;
  position:sticky;bottom:0;z-index:100;}
#wl-add-btn{background:#4f8ef7;color:#fff;border:none;border-radius:5px;
  padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;}
#wl-add-btn:hover{background:#6ba3ff;}
#wl-clear-btn{background:transparent;color:#6b7194;border:1px solid #252a3a;
  border-radius:5px;padding:6px 12px;font-size:12px;cursor:pointer;}
#wl-clear-btn:hover{color:#e8eaf0;}
.cb-cell{width:28px;text-align:center;padding:4px 2px;}
.cb-th{width:28px;text-align:center;}
input.row-check{cursor:pointer;width:14px;height:14px;accent-color:#4f8ef7;}
.wl-toast{position:fixed;bottom:58px;right:20px;background:#1e2538;
  border:1px solid #252a3a;color:#e8eaf0;padding:10px 16px;border-radius:6px;
  font-size:12px;opacity:0;transition:opacity .3s;z-index:10000;max-width:340px;}
.wl-toast.show{opacity:1;}
.wl-toast.warn{border-color:#f0a500;color:#f0a500;}
"""

_WL_BAR = (
    '<div id="wl-bar" style="opacity:.5;pointer-events:none;">'
    '<span id="wl-count">0 selected</span>'
    '<button id="wl-add-btn">+ Add to Deep Dive</button>'
    '<button id="wl-clear-btn">Clear</button>'
    '</div>'
)

_WL_JS_TMPL = r"""
(function(){
var WL_PORT=__PORT__;
function wlTickers(){return[...document.querySelectorAll('input.row-check:checked')].map(function(c){return c.value;});}
function wlUpdate(){
  var n=wlTickers().length,cnt=document.getElementById('wl-count'),bar=document.getElementById('wl-bar');
  if(cnt)cnt.textContent=n+' selected';
  if(bar){bar.style.opacity=n>0?'1':'0.5';bar.style.pointerEvents=n>0?'auto':'none';}
}
function wlToast(msg,warn){
  var t=document.createElement('div');
  t.className='wl-toast'+(warn?' warn':'');
  t.textContent=msg;document.body.appendChild(t);
  setTimeout(function(){t.classList.add('show');},10);
  setTimeout(function(){t.classList.remove('show');setTimeout(function(){t.remove();},400);},3500);
}
document.addEventListener('DOMContentLoaded',function(){
  var allCb=document.getElementById('cb-all');
  if(allCb)allCb.addEventListener('change',function(){
    document.querySelectorAll('input.row-check').forEach(function(c){c.checked=allCb.checked;});
    wlUpdate();
  });
  document.addEventListener('change',function(e){if(e.target.classList.contains('row-check'))wlUpdate();});
  var addBtn=document.getElementById('wl-add-btn');
  if(addBtn)addBtn.addEventListener('click',function(){
    var tickers=wlTickers();if(!tickers.length)return;
    fetch('http://localhost:'+WL_PORT+'/api/watchlist/add',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tickers:tickers})
    }).then(function(r){return r.ok?r.json():Promise.reject('HTTP '+r.status);})
    .then(function(d){
      wlToast('\u2713 '+d.added+' added to deep dive'+(d.skipped?' \u00b7 '+d.skipped+' already present':'')+'  ('+d.total+' total)');
      document.querySelectorAll('input.row-check,#cb-all').forEach(function(c){c.checked=false;});
      wlUpdate();
    }).catch(function(){
      var csv='ticker,shares\n'+tickers.map(function(t){return t+',0';}).join('\n');
      var a=document.createElement('a');
      a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
      a.download='deepDiveTickers_export.csv';document.body.appendChild(a);a.click();a.remove();
      wlToast('Suite offline \u2014 downloaded as CSV',true);
    });
  });
  var clrBtn=document.getElementById('wl-clear-btn');
  if(clrBtn)clrBtn.addEventListener('click',function(){
    document.querySelectorAll('input.row-check,#cb-all').forEach(function(c){c.checked=false;});
    wlUpdate();
  });
  wlUpdate();
});
})();
"""

def _wl_js(port: int) -> str:
    return _WL_JS_TMPL.replace("__PORT__", str(port))

# ── CSV EXPORT ─────────────────────────────────────────────────────────────────
def write_csv(results: list, path: str):
    fields = ["rank","ticker","sector","score","grade",
              "price","mktcap","eps_g","rev_g","roe","roic",
              "n_ratio","p3m","p1m","pe","beta"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(results, 1):
            row = {k: r.get(k, "") for k in fields}
            row["rank"] = i
            w.writerow(row)

# ── HTML ───────────────────────────────────────────────────────────────────────
def build_html(results: list, ts: str, total_in: int,
               index_name: str, suite_port: int = 5050) -> str:

    scored   = [r for r in results if r.get("score") is not None]
    excluded = [r for r in results if r.get("score") is None]
    n_scored = len(scored)

    top5 = scored[:5]
    top5_html = "  ".join(
        f'<b style="color:#4f8ef7">{r["ticker"]}</b>'
        f'<span style="color:#6b7194;font-size:11px"> {r["score"]:.0f}pt [{r["grade"]}]</span>'
        for r in top5
    ) if top5 else '<span style="color:#6b7194">No qualifying stocks</span>'

    all_sectors = sorted({r.get("sector", "") for r in scored if r.get("sector")})
    sector_btns = "".join(
        f'<button class="sbtn" onclick="setSector({repr(s)}, this)">{s}</button>'
        for s in all_sectors
    )

    rows_html = ""
    for i, r in enumerate(scored, 1):
        gc  = _grade_color(r["grade"])
        sc  = _score_color(r["score"])
        eps_c = _pct_color(r.get("eps_g"), lo=0, hi=25)
        roe_c = _pct_color(r.get("roe"),   lo=0, hi=20)
        n_c   = "#00c896" if (r.get("n_ratio") or 0) >= 80 else \
                "#f0a500" if (r.get("n_ratio") or 0) >= 65 else "#e05c5c"
        p3m_c = "#00c896" if (r.get("p3m") or 0) >= 10 else \
                "#f0a500" if (r.get("p3m") or 0) >= 0  else "#e05c5c"
        sec_safe = r.get('sector','').replace("'","")

        rows_html += f"""
<tr data-sector="{sec_safe}">
  <td class="cb-cell"><input type="checkbox" class="row-check" value="{r['ticker']}"></td>
  <td class="num">{i}</td>
  <td class="tk"><b>{r['ticker']}</b></td>
  <td>{r.get('sector','')}</td>
  <td style="color:{sc};font-weight:700">{_fmtnum(r.get('score'))}</td>
  <td><span style="color:{gc};font-weight:700;padding:2px 7px;border:1px solid {gc};border-radius:3px">{r.get('grade','')}</span></td>
  <td style="color:{eps_c};font-weight:600">{_fmtpct(r.get('eps_g'),signed=True)}</td>
  <td>{_fmtpct(r.get('rev_g'),signed=True)}</td>
  <td style="color:{roe_c}">{_fmtpct(r.get('roe'))}</td>
  <td>{_fmtpct(r.get('roic'))}</td>
  <td style="color:{n_c};font-weight:600">{_fmtpct(r.get('n_ratio'))}</td>
  <td style="color:{p3m_c};font-weight:600">{_fmtpct(r.get('p3m'),signed=True)}</td>
  <td>{_fmtpct(r.get('p1m'),signed=True)}</td>
  <td>{_fmtnum(r.get('pe'),1)}</td>
  <td>{_fmtmc(r.get('mktcap'))}</td>
  <td class="num pts">{r.get('pts_c',0)}/{r.get('pts_a',0)}/{r.get('pts_n',0)}/{r.get('pts_s',0)}/{r.get('pts_l',0)}</td>
</tr>"""

    for r in excluded:
        rows_html += f"""
<tr style="opacity:.45">
  <td class="cb-cell"></td>
  <td class="num">—</td>
  <td class="tk">{r['ticker']}</td>
  <td>{r.get('sector','')}</td>
  <td colspan="12" style="color:#6b7194;font-size:11px">Below minimum criteria (EPS growth&lt;0% | 3M perf&lt;0% | &lt;55% of 52w high)</td>
</tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CANSLIM Screener — {index_name}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d0f14;--sf:#14171f;--sf2:#1c2030;--bd:#252a3a;
  --tx:#e8eaf2;--mt:#6b7194;--bl:#4f8ef7;--up:#00c896;--dn:#e05c5c;--wn:#f0a500}}
html,body{{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px}}
a{{color:var(--bl);text-decoration:none}}
.hdr{{background:var(--sf);border-bottom:1px solid var(--bd);padding:14px 24px;
  display:flex;align-items:center;justify-content:space-between;gap:16px}}
.hdr h1{{font-size:18px;font-weight:700}}
.hdr .meta{{color:var(--mt);font-size:12px}}
.top5{{background:var(--sf2);border:1px solid var(--bd);border-radius:8px;
  padding:12px 18px;margin:16px 24px 0;font-size:12px}}
.top5 .lbl{{color:var(--mt);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.controls{{display:flex;gap:10px;padding:12px 24px;align-items:center;flex-wrap:wrap}}
.search{{background:var(--sf);border:1px solid var(--bd);color:var(--tx);
  padding:6px 12px;border-radius:5px;font-size:12px;width:200px}}
.search:focus{{outline:none;border-color:var(--bl)}}
.filter-btn{{background:var(--sf);border:1px solid var(--bd);color:var(--mt);
  padding:5px 14px;border-radius:5px;font-size:12px;cursor:pointer}}
.filter-btn:hover{{color:var(--tx)}}
.cnt{{color:var(--mt);font-size:12px;margin-left:auto}}
.secbar{{display:flex;gap:6px;padding:0 24px 10px;flex-wrap:wrap;align-items:center}}
.secbar .lbl{{color:var(--mt);font-size:11px;white-space:nowrap;margin-right:4px}}
.sbtn{{background:var(--sf);border:1px solid var(--bd);color:var(--mt);
  padding:4px 10px;border-radius:5px;font-size:11px;cursor:pointer}}
.sbtn:hover{{color:var(--tx)}}
.sbtn.active{{border-color:var(--bl);color:var(--bl)}}
.wrap{{padding:0 24px 24px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:var(--sf2);color:var(--mt);padding:8px 10px;text-align:left;
  border-bottom:2px solid var(--bd);cursor:pointer;white-space:nowrap;user-select:none}}
th:hover{{color:var(--tx)}}
th.sort-asc::after{{content:" ▲"}}
th.sort-desc::after{{content:" ▼"}}
td{{padding:7px 10px;border-bottom:1px solid var(--bd)}}
tr:hover td{{background:var(--sf2)}}
.num{{text-align:right}}
.tk{{font-weight:700;color:var(--bl)}}
.pts{{color:var(--mt);font-size:10px;letter-spacing:.3px}}
.legend{{padding:10px 24px;color:var(--mt);font-size:11px;line-height:1.8}}
{_WL_CSS}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>CANSLIM Screener — {index_name}</h1>
    <div class="meta">{n_scored} qualifying stocks &nbsp;·&nbsp; {total_in} scanned &nbsp;·&nbsp; {ts}</div>
  </div>
  <div style="font-size:11px;color:var(--mt);text-align:right">
    Score 100 = ideal CANSLIM setup<br>
    Minimum: EPS growth ≥0% · 3M perf ≥0% · 52w pos ≥55%
  </div>
</div>

<div class="top5">
  <div class="lbl">Top CANSLIM picks</div>
  <div>{top5_html}</div>
</div>

<div class="controls">
  <input class="search" id="search" type="text" placeholder="Search ticker / sector…" oninput="filterTable()">
  <button class="filter-btn" id="btn-all" onclick="setGradeFilter('all')">All</button>
  <button class="filter-btn" id="btn-A"   onclick="setGradeFilter('A')">A+ / A</button>
  <button class="filter-btn" id="btn-B"   onclick="setGradeFilter('B')">B+ / B</button>
  <span class="cnt" id="row-count">{n_scored} stocks</span>
</div>
<div class="secbar">
  <span class="lbl">Sector</span>
  <button class="sbtn all-btn active" onclick="setSector('', this)">All</button>
  {sector_btns}
</div>

<div class="wrap">
<table>
<thead>
<tr>
  <th class="cb-th"><input type="checkbox" id="cb-all" title="Select all"></th>
  <th onclick="sortTable(1)">#</th>
  <th onclick="sortTable(2)">Ticker</th>
  <th onclick="sortTable(3)">Sector</th>
  <th onclick="sortTable(4)">Score</th>
  <th onclick="sortTable(5)">Grade</th>
  <th onclick="sortTable(6)">C: EPS Growth</th>
  <th onclick="sortTable(7)">Rev Growth</th>
  <th onclick="sortTable(8)">A: ROE</th>
  <th onclick="sortTable(9)">ROIC</th>
  <th onclick="sortTable(10)">N: % of 52wH</th>
  <th onclick="sortTable(11)">L: 3M Perf</th>
  <th onclick="sortTable(12)">1M Perf</th>
  <th onclick="sortTable(13)">P/E</th>
  <th onclick="sortTable(14)">Mkt Cap</th>
  <th>Pts C/A/N/S/L</th>
</tr>
</thead>
<tbody id="cs-tbody">
{rows_html}
</tbody>
</table>
</div>

<div class="legend">
  <b>C</b> Current EPS growth (TTM YoY %) ·
  <b>A</b> Annual quality (ROE %) ·
  <b>N</b> Near new high (price as % of 52w high) ·
  <b>S</b> Size score (market cap sweet spot: $1B–$15B = 15 pts) ·
  <b>L</b> Leader (3M price performance %) ·
  <b>Pts</b> = C/A/N/S/L pillar breakdown · Max 100 pts total
</div>

<script>
let _sortCol = -1, _sortAsc = true, _gf = 'all';
const activeSectors = new Set();
function setSector(sec, btn) {{
  if (sec === '') {{
    activeSectors.clear();
  }} else {{
    if (activeSectors.has(sec)) activeSectors.delete(sec);
    else activeSectors.add(sec);
  }}
  document.querySelectorAll('.sbtn:not(.all-btn)').forEach(b =>
    b.classList.toggle('active', activeSectors.has(b.textContent.trim()))
  );
  const allBtn = document.querySelector('.sbtn.all-btn');
  if (allBtn) allBtn.classList.toggle('active', activeSectors.size === 0);
  filterTable();
}}
function sortTable(col) {{
  const tb = document.getElementById('cs-tbody');
  const rows = Array.from(tb.rows);
  if (_sortCol === col) _sortAsc = !_sortAsc;
  else {{ _sortCol = col; _sortAsc = true; }}
  document.querySelectorAll('th').forEach((t,i) => {{
    t.classList.remove('sort-asc','sort-desc');
    if (i === col) t.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
  }});
  rows.sort((a,b) => {{
    let av = a.cells[col]?.textContent?.trim() ?? '';
    let bv = b.cells[col]?.textContent?.trim() ?? '';
    const an = parseFloat(av.replace(/[^0-9.-]/g,'')), bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
    if (!isNaN(an) && !isNaN(bn)) return _sortAsc ? an-bn : bn-an;
    return _sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tb.appendChild(r));
}}
function setGradeFilter(g) {{
  _gf = g;
  document.querySelectorAll('.filter-btn').forEach(b => b.style.borderColor = '');
  const btn = document.getElementById('btn-'+g);
  if (btn) {{ btn.style.borderColor='var(--bl)'; btn.style.color='var(--bl)'; }}
  filterTable();
}}
function filterTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#cs-tbody tr');
  let shown = 0;
  rows.forEach(row => {{
    const tk = row.cells[2]?.textContent?.toLowerCase() ?? '';
    const sc = row.cells[3]?.textContent?.toLowerCase() ?? '';
    const gr = row.cells[5]?.textContent?.trim() ?? '';
    const mq = !q || tk.includes(q) || sc.includes(q);
    const ms = activeSectors.size === 0 || activeSectors.has(row.dataset.sector ?? '');
    const mg = _gf==='all'
      || (_gf==='A' && (gr==='A+' || gr==='A'))
      || (_gf==='B' && (gr==='B+' || gr==='B'));
    row.style.display = (mq && ms && mg) ? '' : 'none';
    if (mq && ms && mg) shown++;
  }});
  document.getElementById('row-count').textContent = shown + ' stocks';
}}
</script>
<script>{_wl_js(suite_port)}</script>
{_WL_BAR}
</body>
</html>"""


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--index",     default="SPX")
    ap.add_argument("--top",       type=int,   default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--csv",       action="store_true")
    ap.add_argument("--help",      action="store_true")
    args = ap.parse_args()

    if args.help:
        print(__doc__); sys.exit(0)

    index_code = args.index.upper()
    if index_code not in INDEX_CONFIG:
        print(f"ERROR: Unknown index '{index_code}'. Valid: {', '.join(INDEX_CONFIG)}")
        sys.exit(1)

    print("=" * 60)
    print("  CANSLIM Screener")
    print("=" * 60)

    print("\n[1/4] Fetching stocks from TradingView...")
    df = fetch_stocks(index_code, limit=args.top)
    total_in = len(df)
    print(f"  Total fetched: {total_in}")

    print("\n[2/4] Parsing rows...")
    rows, skipped = [], 0
    for _, row in df.iterrows():
        try:
            d = parse_row(row.to_dict())
            if d["ticker"]: rows.append(d)
            else: skipped += 1
        except Exception:
            skipped += 1
    print(f"  Parsed: {len(rows)} | Skipped (no price): {skipped}")

    print("\n[3/4] Scoring CANSLIM factors...")
    scored_all, below_min = [], []
    for r in rows:
        if r.get("sector") in EXCLUDE_SECTORS:
            continue
        result = score_stock(r)
        if result is None:
            below_min.append(r)
        else:
            scored_all.append(result)

    scored_all.sort(key=lambda x: x["score"], reverse=True)

    min_sc = args.min_score
    display = [r for r in scored_all if r["score"] >= min_sc]
    print(f"  Qualifying: {len(scored_all)} | Below min-score ({min_sc}): "
          f"{len(scored_all)-len(display)} | Sector-excluded/below-gate: {len(below_min)}")

    print("\n  Top 10 CANSLIM picks:")
    for r in display[:10]:
        print(f"    {r['ticker']:<6} score={r['score']:.0f} [{r['grade']}]  "
              f"EPS={_fmtpct(r.get('eps_g'),signed=True):<8}  "
              f"3M={_fmtpct(r.get('p3m'),signed=True):<8}  "
              f"52wH={_fmtpct(r.get('n_ratio'))}")

    print("\n[4/4] Building HTML report...")
    ts         = datetime.datetime.now().strftime("%b %d, %Y  %H:%M")
    suite_port = int(os.environ.get("VALUATION_SUITE_PORT", "5050"))
    all_results = display + below_min
    html = build_html(all_results, ts, total_in,
                      INDEX_CONFIG[index_code]["name"], suite_port=suite_port)

    index_slug = index_code.lower()
    date_str   = datetime.datetime.now().strftime("%Y_%m_%d")
    out_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canslimData")
    os.makedirs(out_dir, exist_ok=True)
    out     = os.path.join(out_dir, f"{date_str}_canslim_{index_slug}.html")
    out_csv = os.path.join(out_dir, f"{date_str}_canslim_{index_slug}.csv")

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {out}")

    if args.csv:
        write_csv(display, out_csv)
        print(f"  CSV  saved: {out_csv}")

    if not os.environ.get("VALUATION_SUITE_LAUNCHED"):
        webbrowser.open("file://" + os.path.abspath(out))
    print(f"\nDone. ({len(display)} stocks qualifying)")
    print("=" * 60)


if __name__ == "__main__":
    main()
