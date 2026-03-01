#!/usr/bin/env python3
"""
ValuationSuite — Flask Web Launcher
=====================================
Browser-based replacement for the Tkinter launcher.

Usage
-----
    python app.py              # starts server + opens browser
    python app.py --port 5050  # custom port
    python app.py --no-browser # skip auto-open
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime

from flask import Flask, Response, jsonify, request

ROOT = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Script catalogue ──────────────────────────────────────────────────────────

SCRIPTS = {
    "Macro Dashboard": {
        "path": os.path.join(ROOT, "7_macroDashboard", "macroDashboard.py"),
        "desc": "Macro regime dashboard — rates, yield curve, volatility, credit, sector rotation & cross-asset performance.",
        "icon": "🌐",
        "params": [],
    },
    "Value Screener": {
        "path": os.path.join(ROOT, "1_valueScreener", "valueScreener.py"),
        "desc": "Screen S&P 500, Nasdaq 100, Russell 2000, or TSX for undervalued stocks.",
        "icon": "🔍",
        "params": [
            dict(id="index", label="Index",      type="option", flag="--index",
                 required=True,  default="SPX",  values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",   label="Top N",      type="entry",  flag="--top",
                 required=False, default="50"),
            dict(id="csv",   label="Export CSV", type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Growth Screener": {
        "path": os.path.join(ROOT, "2_growthScreener", "growthScreener.py"),
        "desc": "Screen indices for high-growth stocks using a 4-pillar quality + momentum model.",
        "icon": "📈",
        "params": [
            dict(id="index", label="Index",      type="option", flag="--index",
                 required=True,  default="NDX",  values=["SPX", "NDX", "RUT", "TSX"]),
            dict(id="top",   label="Top N",      type="entry",  flag="--top",
                 required=False, default="50"),
            dict(id="csv",   label="Export CSV", type="check",  flag="--csv",
                 required=False, default=True),
        ],
    },
    "Valuation Master": {
        "path": os.path.join(ROOT, "3_valuationTool", "valuationMaster.py"),
        "desc": "Deep single-stock valuation — 5-tab HTML report with 20+ methods.",
        "icon": "📊",
        "params": [
            dict(id="ticker",   label="Ticker",        type="entry",  positional=True,
                 required=True,  default="NVDA"),
            dict(id="backtest", label="Backtest days", type="entry",  flag="--backtest",
                 required=False, default="90"),
            dict(id="wacc",     label="WACC override", type="entry",  flag="--wacc",
                 required=False, default=""),
        ],
    },
    "Run Model": {
        "path": os.path.join(ROOT, "run_model.py"),
        "desc": "Run any single valuation model on a ticker with optional backtest plot.",
        "icon": "🔬",
        "params": [
            dict(id="ticker", label="Ticker",          type="entry",  positional=True,
                 required=True,  default="NVDA"),
            dict(id="model",  label="Model",           type="option", positional=True,
                 required=True,  default="dcf",
                 values=["dcf", "three-stage", "monte-carlo", "pfcf", "pe", "ev-ebitda",
                         "fcf-yield", "rim", "roic", "ncav", "mean-reversion",
                         "reverse-dcf", "peg", "ev-ntm", "tam", "rule40", "erg",
                         "scurve", "pie", "ddm", "graham", "multifactor", "bayesian", "all"]),
            dict(id="wacc",   label="WACC override",   type="entry",  flag="--wacc",
                 required=False, default=""),
            dict(id="growth", label="Growth override", type="entry",  flag="--growth",
                 required=False, default=""),
            dict(id="plot",   label="Plot backtest",   type="check",  flag="--plot",
                 required=False, default=False),
            dict(id="days",   label="Plot days",       type="entry",  flag="--days",
                 required=False, default="1000"),
        ],
    },
    "Sentiment Analyzer": {
        "path": os.path.join(ROOT, "6_sentimentAnalyzer", "sentimentAnalyzer.py"),
        "desc": "Multi-source sentiment — news, social, short interest, insider activity, options, analyst ratings, SEC filings & congressional trading.",
        "icon": "📡",
        "params": [
            dict(id="csvfile", label="Portfolio CSV", type="file",  positional=True,
                 required=True,
                 default=os.path.join(ROOT, "4_portfolioAnalyzer", "FilPortfolio.csv")),
            dict(id="days",    label="Lookback days", type="entry", flag="--days",
                 required=False, default="30"),
        ],
    },
    "Portfolio Analyzer": {
        "path": os.path.join(ROOT, "4_portfolioAnalyzer", "portfolioAnalyzer.py"),
        "desc": "Multi-stock portfolio dashboard — signals, technicals, rebalancing, backtesting.",
        "icon": "💼",
        "params": [
            dict(id="csvfile",  label="Portfolio CSV", type="file",  positional=True,
                 required=True,
                 default=os.path.join(ROOT, "4_portfolioAnalyzer", "FilPortfolio.csv")),
            dict(id="backtest", label="Backtest days", type="entry", flag="--backtest",
                 required=False, default="90"),
        ],
    },
}

# Sidebar group layout — defines sections and order shown in the UI
SIDEBAR_GROUPS = [
    {"label": "MACRO TRENDS",      "scripts": ["Macro Dashboard"]},
    {"label": "SCREENERS",         "scripts": ["Value Screener", "Growth Screener"]},
    {"label": "VALUATION",         "scripts": ["Valuation Master", "Run Model"]},
    {"label": "PORTFOLIO ANALYSIS","scripts": ["Sentiment Analyzer", "Portfolio Analyzer"]},
]

# ── Active runs ───────────────────────────────────────────────────────────────
# run_id → {"proc": Popen, "q": Queue, "done": bool, "tmpfile": path|None}
_runs: dict = {}
_runs_lock   = threading.Lock()


def _reader_thread(run_id: str) -> None:
    """Read subprocess stdout into the run's queue; put None sentinel when done."""
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        return

    proc = run["proc"]
    q    = run["q"]
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
        proc.stdout.close()
        proc.wait()
        rc = proc.returncode
        sep = "─" * 54
        q.put(f"\n{sep}\n")
        q.put("  ✓  Completed (exit 0)\n" if rc == 0 else f"  ✗  Exited with code {rc}\n")
        q.put(f"{sep}\n")
    except Exception as exc:
        q.put(f"\nERROR: {exc}\n")
    finally:
        q.put(None)  # sentinel
        with _runs_lock:
            if run_id in _runs:
                _runs[run_id]["done"] = True
            # Clean up any temp uploaded file
            tmp = _runs.get(run_id, {}).get("tmpfile")
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/api/scripts")
def api_scripts():
    return jsonify(SCRIPTS)


@app.route("/api/groups")
def api_groups():
    return jsonify(SIDEBAR_GROUPS)


@app.route("/api/run", methods=["POST"])
def api_run():
    data        = request.get_json(force=True) or {}
    script_name = data.get("script", "")
    params      = data.get("params", {})

    if script_name not in SCRIPTS:
        return jsonify({"error": f"Unknown script: {script_name}"}), 400

    info = SCRIPTS[script_name]
    cmd  = [sys.executable, "-u", info["path"]]  # -u = unbuffered stdout
    cwd  = os.path.dirname(info["path"])

    for p in info["params"]:
        pid   = p["id"]
        ptype = p["type"]
        flag  = p.get("flag")
        val   = params.get(pid, p.get("default", ""))

        if ptype == "check":
            if val and flag:
                cmd.append(flag)
        elif ptype == "file":
            val = str(val).strip()
            if val:
                cmd.append(val)
        else:
            val = str(val).strip()
            if not val:
                continue
            if flag:
                cmd.extend([flag, val])
            else:
                cmd.append(val)

    run_id = uuid.uuid4().hex[:8]
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            start_new_session=True,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    q = queue.Queue()
    with _runs_lock:
        _runs[run_id] = {"proc": proc, "q": q, "done": False, "tmpfile": None}

    threading.Thread(target=_reader_thread, args=(run_id,), daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/api/stream/<run_id>")
def api_stream(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        return Response("data: [DONE]\n\n", mimetype="text/event-stream")

    def generate():
        q = run["q"]
        while True:
            try:
                line = q.get(timeout=25)
            except queue.Empty:
                yield ": keep-alive\n\n"  # prevent proxy timeouts
                continue
            if line is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(line)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stop/<run_id>", methods=["POST"])
def api_stop(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if run:
        proc = run["proc"]
        if proc.poll() is None:
            proc.terminate()
    return jsonify({"ok": True})


@app.route("/api/exit", methods=["POST"])
def api_exit():
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Save an uploaded portfolio CSV to the portfolioAnalyzer directory."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received"}), 400
    safe_name = os.path.basename(f.filename)
    dest = os.path.join(ROOT, "4_portfolioAnalyzer", safe_name)
    f.save(dest)
    return jsonify({"path": dest})


# ── Embedded HTML / CSS / JS ───────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Valuation Suite</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0e1117;
    --panel:   #161b22;
    --card:    #1c2128;
    --border:  #30363d;
    --accent:  #4f8ef7;
    --green:   #00c896;
    --muted:   #8b949e;
    --text:    #e6edf3;
    --btn-run: #1f6feb;
    --btn-stp: #da3633;
    --console: #060d17;
    --yellow:  #f0a500;
    --red:     #f85149;
    --radius:  8px;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text);
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

  /* ── Layout ── */
  .shell   { display: flex; flex-direction: column; height: 100vh; }
  .topbar  { display: flex; align-items: center; gap: 10px;
             background: var(--panel); border-bottom: 1px solid var(--border);
             padding: 0 20px; height: 52px; flex-shrink: 0; }
  .topbar h1 { font-size: 17px; font-weight: 700; color: var(--text); flex: 1; }
  .btn-exit { font-size: 12px; font-weight: 600; padding: 6px 16px; border: 1px solid var(--border);
              border-radius: 6px; background: transparent; color: var(--muted); cursor: pointer;
              transition: background .15s, color .15s, border-color .15s; }
  .btn-exit:hover { background: var(--btn-stp); border-color: var(--btn-stp); color: #fff; }
  .body    { display: flex; flex: 1; overflow: hidden; }

  /* ── Sidebar ── */
  .sidebar { width: 190px; background: var(--panel);
             border-right: 1px solid var(--border);
             display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto; }
  .sidebar-label { font-size: 10px; font-weight: 700; color: var(--muted);
                   letter-spacing: .08em; padding: 18px 16px 6px; }
  .sidebar hr { border: none; border-top: 1px solid var(--border); margin: 0 16px 8px; }
  .sbtn { display: block; width: 100%; text-align: left;
          background: transparent; border: none; color: var(--muted);
          font-size: 13px; padding: 9px 14px; cursor: pointer; border-radius: 6px;
          margin: 1px 6px; width: calc(100% - 12px); transition: background .15s, color .15s; }
  .sbtn:hover   { background: var(--card); color: var(--text); }
  .sbtn.active  { background: var(--card); color: var(--text); font-weight: 600; }

  /* ── Content ── */
  .content { flex: 1; display: flex; flex-direction: column; overflow: hidden; padding: 14px; gap: 10px; }

  /* ── Params card ── */
  .params-card { background: var(--card); border: 1px solid var(--border);
                 border-radius: var(--radius); padding: 14px 18px 10px; flex-shrink: 0; }
  .params-card h2 { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: var(--text); }
  .params-grid { display: flex; flex-wrap: wrap; gap: 10px 24px; align-items: center; }
  .param-group { display: flex; align-items: center; gap: 8px; }
  .param-group label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .param-group label.req { color: var(--text); }

  input[type="text"] {
    background: #0d1117; border: 1px solid var(--border); color: var(--text);
    font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
    padding: 5px 9px; border-radius: 5px; width: 130px; outline: none;
    transition: border-color .15s;
  }
  input[type="text"]:focus { border-color: var(--accent); }

  select {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
    padding: 5px 9px; border-radius: 5px; outline: none; cursor: pointer;
  }

  .toggle { font-size: 12px; padding: 5px 12px; border-radius: 5px;
            border: none; cursor: pointer; font-weight: 600; transition: background .15s; }
  .toggle.on  { background: var(--accent); color: #fff; }
  .toggle.off { background: var(--border); color: var(--muted); }

  .file-row { display: flex; align-items: center; gap: 8px; }
  .file-row input[type="text"] { width: 280px; }
  .file-row input[type="file"] { display: none; }
  .browse-btn { font-size: 12px; padding: 5px 12px; border-radius: 5px;
                border: 1px solid var(--border); background: var(--border);
                color: var(--text); cursor: pointer; white-space: nowrap; }
  .browse-btn:hover { background: var(--accent); border-color: var(--accent); }

  /* ── Description ── */
  .desc { font-size: 12px; color: var(--muted); flex-shrink: 0; }

  /* ── Button row ── */
  .btn-row { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .btn { font-size: 13px; font-weight: 700; padding: 8px 22px; border: none;
         border-radius: 6px; cursor: pointer; transition: opacity .15s; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-run { background: var(--btn-run); color: #fff; }
  .btn-run:not(:disabled):hover { background: #388bfd; }
  .btn-stp { background: var(--btn-stp); color: #fff; }
  .btn-stp:not(:disabled):hover { background: var(--red); }
  .btn-clr { background: var(--card); color: var(--muted); font-weight: 400; margin-left: auto; }
  .btn-clr:hover { background: var(--border); color: var(--text); }

  /* ── Console ── */
  .console-wrap { flex: 1; background: var(--card); border: 1px solid var(--border);
                  border-radius: var(--radius); display: flex; flex-direction: column;
                  overflow: hidden; min-height: 0; }
  .console-label { font-size: 10px; font-weight: 700; color: var(--muted);
                   letter-spacing: .08em; padding: 8px 12px 4px; flex-shrink: 0; }
  #console { flex: 1; background: var(--console); color: var(--green); overflow-y: auto;
             font-family: "Menlo","Monaco","Courier New", monospace; font-size: 12px;
             line-height: 1.55; padding: 10px 14px; white-space: pre-wrap; word-break: break-word; }

  .c-err  { color: var(--red); }
  .c-warn { color: var(--yellow); }
  .c-ok   { color: var(--green); }
  .c-info { color: var(--accent); }
  .c-hdr  { color: var(--text); font-weight: bold; }
  .c-dim  { color: var(--muted); }
</style>
</head>
<body>
<div class="shell">

  <!-- Top bar -->
  <div class="topbar">
    <h1>📊&nbsp; Valuation Suite</h1>
    <button class="btn-exit" onclick="exitApp()">Exit</button>
  </div>

  <div class="body">

    <!-- Sidebar -->
    <nav class="sidebar" id="sidebar"></nav>

    <!-- Main content -->
    <div class="content">
      <div class="params-card">
        <h2 id="script-title">—</h2>
        <div class="params-grid" id="params-grid"></div>
      </div>

      <p class="desc" id="script-desc"></p>

      <div class="btn-row">
        <button class="btn btn-run" id="btn-run" onclick="runScript()">▶&nbsp; Run</button>
        <button class="btn btn-stp" id="btn-stp" onclick="stopScript()" disabled>■&nbsp; Stop</button>
        <button class="btn btn-clr" onclick="clearConsole()">Clear</button>
      </div>

      <div class="console-wrap">
        <div class="console-label">OUTPUT</div>
        <div id="console"></div>
      </div>
    </div>
  </div>
</div>

<script>
let SCRIPTS = {};
let currentScript = null;
let currentRunId  = null;
let currentES     = null;   // EventSource

// ── Bootstrap ──────────────────────────────────────────────────────────────
Promise.all([
  fetch("/api/scripts").then(r => r.json()),
  fetch("/api/groups").then(r => r.json()),
]).then(([scripts, groups]) => {
  SCRIPTS = scripts;
  const sidebar = document.getElementById("sidebar");

  groups.forEach((group, gi) => {
    if (gi > 0) {
      const hr = document.createElement("hr");
      sidebar.appendChild(hr);
    }
    const lbl = document.createElement("div");
    lbl.className = "sidebar-label";
    lbl.textContent = group.label;
    sidebar.appendChild(lbl);

    group.scripts.forEach(name => {
      if (!SCRIPTS[name]) return;
      const b = document.createElement("button");
      b.className   = "sbtn";
      b.textContent = SCRIPTS[name].icon + "  " + name;
      b.onclick     = () => selectScript(name);
      b.id          = "sbtn-" + name;
      sidebar.appendChild(b);
    });
  });

  // Select first script of first group by default
  const firstName = groups[0]?.scripts[0];
  if (firstName && SCRIPTS[firstName]) selectScript(firstName);
});

// ── Script selection ───────────────────────────────────────────────────────
function selectScript(name) {
  currentScript = name;
  const info = SCRIPTS[name];

  document.querySelectorAll(".sbtn").forEach(b => b.classList.remove("active"));
  const activeBtn = document.getElementById("sbtn-" + name);
  if (activeBtn) activeBtn.classList.add("active");

  document.getElementById("script-title").textContent = info.icon + "  " + name;
  document.getElementById("script-desc").textContent  = info.desc;

  const grid = document.getElementById("params-grid");
  grid.innerHTML = "";

  info.params.forEach(p => {
    const group = document.createElement("div");
    group.className = "param-group";

    const lbl = document.createElement("label");
    lbl.textContent = p.label + (p.required ? " *" : "");
    if (p.required) lbl.classList.add("req");
    group.appendChild(lbl);

    if (p.type === "entry") {
      const inp = document.createElement("input");
      inp.type  = "text";
      inp.id    = "param-" + p.id;
      inp.value = p.default || "";
      group.appendChild(inp);

    } else if (p.type === "option") {
      const sel = document.createElement("select");
      sel.id = "param-" + p.id;
      (p.values || []).forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        if (v === p.default) opt.selected = true;
        sel.appendChild(opt);
      });
      group.appendChild(sel);

    } else if (p.type === "check") {
      const btn = document.createElement("button");
      btn.className = "toggle " + (p.default ? "on" : "off");
      btn.id        = "param-" + p.id;
      btn.textContent = p.default ? "● ON" : "○ OFF";
      btn.dataset.val = p.default ? "1" : "0";
      btn.onclick = () => {
        const on = btn.dataset.val === "0";
        btn.dataset.val   = on ? "1" : "0";
        btn.className     = "toggle " + (on ? "on" : "off");
        btn.textContent   = on ? "● ON" : "○ OFF";
      };
      group.appendChild(btn);

    } else if (p.type === "file") {
      const row = document.createElement("div");
      row.className = "file-row";

      const inp = document.createElement("input");
      inp.type  = "text";
      inp.id    = "param-" + p.id;
      inp.value = p.default || "";

      const fileInp = document.createElement("input");
      fileInp.type  = "file";
      fileInp.accept = ".csv";
      fileInp.id    = "fileinput-" + p.id;
      fileInp.onchange = () => {
        const file = fileInp.files[0];
        if (!file) return;
        const fd = new FormData();
        fd.append("file", file);
        fetch("/api/upload", { method: "POST", body: fd })
          .then(r => r.json())
          .then(data => {
            if (data.path) inp.value = data.path;
            else logLine("  ⚠  Upload failed: " + (data.error || "unknown"), "warn");
          });
      };

      const browseBtn = document.createElement("button");
      browseBtn.className   = "browse-btn";
      browseBtn.textContent = "Browse…";
      browseBtn.onclick     = () => fileInp.click();

      row.appendChild(inp);
      row.appendChild(fileInp);
      row.appendChild(browseBtn);
      group.appendChild(row);
    }

    grid.appendChild(group);
  });
}

// ── Build params dict from current form ───────────────────────────────────
function collectParams() {
  const info   = SCRIPTS[currentScript];
  const params = {};
  for (const p of info.params) {
    if (p.type === "check") {
      const btn = document.getElementById("param-" + p.id);
      params[p.id] = btn ? btn.dataset.val === "1" : false;
    } else {
      const el = document.getElementById("param-" + p.id);
      params[p.id] = el ? el.value : "";
    }
  }
  return params;
}

// ── Run ────────────────────────────────────────────────────────────────────
function runScript() {
  if (currentRunId) return;

  const info   = SCRIPTS[currentScript];
  const params = collectParams();

  // Validate required
  for (const p of info.params) {
    if (p.required) {
      const v = params[p.id];
      if (!v || !String(v).trim()) {
        logLine("  ⚠  '" + p.label + "' is required.\n", "warn");
        return;
      }
    }
  }

  clearConsole();
  const ts = new Date().toLocaleTimeString();
  logLine("─".repeat(54) + "\n", "dim");
  logLine("  " + info.icon + "   " + currentScript + "   ·   " + ts + "\n", "hdr");
  logLine("─".repeat(54) + "\n\n", "dim");

  setRunning(true);

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ script: currentScript, params }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      logLine("  ERROR: " + data.error + "\n", "err");
      setRunning(false);
      return;
    }
    currentRunId = data.run_id;
    startStream(currentRunId);
  })
  .catch(err => {
    logLine("  ERROR: " + err + "\n", "err");
    setRunning(false);
  });
}

// ── SSE stream ─────────────────────────────────────────────────────────────
function startStream(runId) {
  if (currentES) { currentES.close(); currentES = null; }
  currentES = new EventSource("/api/stream/" + runId);
  currentES.onmessage = e => {
    if (e.data === "[DONE]") {
      currentES.close();
      currentES   = null;
      currentRunId = null;
      setRunning(false);
      return;
    }
    const line = JSON.parse(e.data);
    logLine(line, tagLine(line));
  };
  currentES.onerror = () => {
    currentES.close();
    currentES   = null;
    currentRunId = null;
    setRunning(false);
  };
}

// ── Stop ───────────────────────────────────────────────────────────────────
function stopScript() {
  if (!currentRunId) return;
  fetch("/api/stop/" + currentRunId, { method: "POST" });
  if (currentES) { currentES.close(); currentES = null; }
  logLine("\n  [Stopped by user]\n", "warn");
  currentRunId = null;
  setRunning(false);
}

// ── Button state ───────────────────────────────────────────────────────────
function setRunning(on) {
  document.getElementById("btn-run").disabled = on;
  document.getElementById("btn-stp").disabled = !on;
}

// ── Console helpers ────────────────────────────────────────────────────────
function tagLine(line) {
  const lo = line.toLowerCase();
  if (/error|traceback|exception|✗/.test(lo)) return "err";
  if (/warning|⚠|skipped|skip/.test(lo))      return "warn";
  if (/✓|saved →|complete| ok|done/.test(lo)) return "ok";
  if (/^\s*\[/.test(line))                     return "info";
  if (/^[╔║╚═]/.test(line))                   return "hdr";
  return "ok";
}

function logLine(text, cls) {
  const con  = document.getElementById("console");
  const span = document.createElement("span");
  span.className   = "c-" + (cls || "ok");
  span.textContent = text;
  con.appendChild(span);
  con.scrollTop = con.scrollHeight;
}

function clearConsole() {
  document.getElementById("console").innerHTML = "";
}

// ── Exit ───────────────────────────────────────────────────────────────────
function exitApp() {
  if (currentRunId) stopScript();
  logLine("\n  Shutting down server…\n", "warn");
  fetch("/api/exit", { method: "POST" })
    .catch(() => {})
    .finally(() => {
      setTimeout(() => window.close(), 400);
    });
}
</script>
</body>
</html>
"""

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ValuationSuite web launcher")
    parser.add_argument("--port",       type=int,  default=5050, help="Port (default 5050)")
    parser.add_argument("--no-browser", action="store_true",     help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=[url]).start()

    print(f"\n  ValuationSuite  →  {url}\n  Ctrl-C to quit\n")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
