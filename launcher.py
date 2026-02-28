#!/usr/bin/env python3
"""
ValuationSuite Launcher
=======================
macOS GUI launcher for the valuationScripts toolkit.

Usage
-----
  python3 launcher.py

To create a double-clickable .app bundle:
  bash create_app.sh
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog
import subprocess, threading, queue, os, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#0e1117"
PANEL   = "#161b22"
CARD    = "#1c2128"
BORDER  = "#30363d"
ACCENT  = "#4f8ef7"
GREEN   = "#00c896"
MUTED   = "#8b949e"
TEXT    = "#e6edf3"
BTN_RUN = "#1f6feb"
BTN_STP = "#da3633"
CONSOLE = "#060d17"
YELLOW  = "#f0a500"
RED     = "#f85149"

# ── Script catalogue ────────────────────────────────────────────────────────────
SCRIPTS = {
    "Valuation Master": {
        "path": os.path.join(ROOT, "3_valuationTool", "valuationMaster.py"),
        "desc": "Deep single-stock valuation — generates a 5-tab HTML report with 20+ methods.",
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
        "desc": "Run any single valuation model on a ticker — prints results + optional backtest plot.",
        "icon": "🔬",
        "params": [
            dict(id="ticker",  label="Ticker",          type="entry",  positional=True,
                 required=True,  default="NVDA"),
            dict(id="model",   label="Model",           type="option", positional=True,
                 required=True,  default="dcf",
                 values=["dcf", "three-stage", "monte-carlo", "pfcf", "pe", "ev-ebitda",
                         "fcf-yield", "rim", "roic", "ncav", "mean-reversion",
                         "reverse-dcf", "peg", "ev-ntm", "tam", "rule40", "erg",
                         "scurve", "pie", "ddm", "graham", "multifactor", "bayesian", "all"]),
            dict(id="wacc",    label="WACC override",   type="entry",  flag="--wacc",
                 required=False, default=""),
            dict(id="growth",  label="Growth override", type="entry",  flag="--growth",
                 required=False, default=""),
            dict(id="plot",    label="Plot backtest",   type="check",  flag="--plot",
                 required=False, default=False),
            dict(id="days",    label="Plot days",       type="entry",  flag="--days",
                 required=False, default="1000"),
        ],
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
    "Portfolio Analyzer": {
        "path": os.path.join(ROOT, "4_portfolioAnalyzer", "portfolioAnalyzer.py"),
        "desc": "Multi-stock portfolio dashboard — signals, technicals, rebalancing, and backtesting.",
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


# ── Main application ────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Valuation Suite")
        self.geometry("1080x740")
        self.minsize(860, 600)
        self.configure(bg=BG)

        self._proc    = None
        self._running = False   # explicit flag — more reliable than proc.poll()
        self._q       = queue.Queue()
        self._pwids   = {}      # param_id → tk variable
        self._cur     = None    # currently selected script name
        self._log_buf = []      # (text, tag) pairs — flushed once per poll cycle

        self._build_ui()
        self._select(list(SCRIPTS)[0])
        self._poll()

    # ── UI construction ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self, bg=PANEL, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="   📊  Valuation Suite",
                 font=("Helvetica Neue", 17, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left", pady=10)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────────────────
        self._sb = tk.Frame(body, bg=PANEL, width=170)
        self._sb.pack(side="left", fill="y")
        self._sb.pack_propagate(False)

        tk.Label(self._sb, text="SCRIPTS",
                 font=("Helvetica Neue", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(pady=(16, 6), padx=14, anchor="w")
        tk.Frame(self._sb, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(0, 8))

        self._sbtn = {}
        for name, info in SCRIPTS.items():
            b = tk.Button(
                self._sb,
                text=f"  {info['icon']}  {name}",
                font=("Helvetica Neue", 12),
                bg=PANEL, fg=MUTED,
                activebackground=CARD, activeforeground=TEXT,
                relief="flat", anchor="w", padx=10, pady=8,
                cursor="hand2",
                command=lambda n=name: self._select(n),
            )
            b.pack(fill="x", padx=6, pady=2)
            self._sbtn[name] = b

        # ── Content area ─────────────────────────────────────────────────────
        content = tk.Frame(body, bg=BG)
        content.pack(side="left", fill="both", expand=True)

        # Parameters card
        self._pcard = tk.Frame(content, bg=CARD)
        self._pcard.pack(fill="x", padx=14, pady=(14, 0))

        # Description
        self._desc_var = tk.StringVar()
        tk.Label(content, textvariable=self._desc_var,
                 font=("Helvetica Neue", 11), bg=BG, fg=MUTED,
                 anchor="w", wraplength=750, justify="left"
                 ).pack(fill="x", padx=14, pady=(6, 4))

        # Buttons row
        brow = tk.Frame(content, bg=BG)
        brow.pack(fill="x", padx=14, pady=(2, 8))

        self._run_b = tk.Button(
            brow, text="▶   Run",
            font=("Helvetica Neue", 13, "bold"),
            bg=BTN_RUN, fg="white", activebackground="#388bfd",
            relief="flat", cursor="hand2", padx=24, pady=8,
            command=self._run,
        )
        self._run_b.pack(side="left", padx=(0, 8))

        self._stp_b = tk.Button(
            brow, text="■  Stop",
            font=("Helvetica Neue", 12),
            bg=BTN_STP, fg="white", activebackground=RED,
            relief="flat", cursor="hand2", padx=16, pady=8,
            state="disabled",
            command=self._stop,
        )
        self._stp_b.pack(side="left", padx=(0, 8))

        tk.Button(
            brow, text="Clear",
            font=("Helvetica Neue", 11),
            bg=CARD, fg=MUTED, activebackground=BORDER,
            relief="flat", cursor="hand2", padx=14, pady=8,
            command=self._clear,
        ).pack(side="right")

        # Console
        cf = tk.Frame(content, bg=CARD)
        cf.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        tk.Label(cf, text=" OUTPUT",
                 font=("Helvetica Neue", 9, "bold"),
                 bg=CARD, fg=MUTED).pack(anchor="w", pady=(8, 2))

        self._con = scrolledtext.ScrolledText(
            cf,
            font=("Menlo", 11),
            bg=CONSOLE, fg=GREEN,
            insertbackground=GREEN, selectbackground=BORDER,
            relief="flat", bd=0, padx=12, pady=8,
            state="disabled", wrap="word",
        )
        self._con.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Console colour tags
        for tag, color, bold in [
            ("err",  RED,    False),
            ("warn", YELLOW, False),
            ("ok",   GREEN,  False),
            ("info", ACCENT, False),
            ("hdr",  TEXT,   True),
            ("dim",  MUTED,  False),
        ]:
            self._con.tag_configure(
                tag, foreground=color,
                font=("Menlo", 11, "bold" if bold else "normal"),
            )

    # ── Script selection ────────────────────────────────────────────────────────
    def _select(self, name):
        self._cur = name
        info = SCRIPTS[name]

        # Highlight sidebar button
        for n, b in self._sbtn.items():
            b.configure(
                bg=CARD if n == name else PANEL,
                fg=TEXT if n == name else MUTED,
            )

        self._desc_var.set(info["desc"])

        # Rebuild parameter widgets
        for w in self._pcard.winfo_children():
            w.destroy()
        self._pwids = {}

        # Script title inside the card
        tk.Label(
            self._pcard,
            text=f"{info['icon']}   {name}",
            font=("Helvetica Neue", 13, "bold"),
            bg=CARD, fg=TEXT,
        ).grid(row=0, column=0, columnspan=6, sticky="w",
               padx=16, pady=(12, 6))

        col = 0
        row = 1
        for p in info["params"]:
            pid, ptype = p["id"], p["type"]
            req   = p.get("required", False)
            label = p["label"] + (" *" if req else "")

            tk.Label(
                self._pcard, text=label,
                font=("Helvetica Neue", 11),
                bg=CARD,
                fg=TEXT if req else MUTED,
            ).grid(row=row, column=col * 3,
                   sticky="w", padx=(16 if col == 0 else 12, 4), pady=7)

            if ptype == "entry":
                var = tk.StringVar(value=p.get("default", ""))
                tk.Entry(
                    self._pcard, textvariable=var,
                    font=("Menlo", 11),
                    bg="#0d1117", fg=TEXT, insertbackground=TEXT,
                    relief="flat", bd=0, width=14,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                ).grid(row=row, column=col * 3 + 1,
                       sticky="w", padx=(0, 10), pady=7)
                self._pwids[pid] = var

            elif ptype == "option":
                var = tk.StringVar(value=p.get("default", ""))
                om = tk.OptionMenu(self._pcard, var, *p["values"])
                om.configure(
                    bg=CARD, fg=TEXT,
                    activebackground=BORDER, activeforeground=TEXT,
                    relief="flat", font=("Menlo", 11),
                    highlightthickness=0, bd=0,
                    padx=8, pady=3,
                )
                om["menu"].configure(
                    bg=CARD, fg=TEXT,
                    activebackground=ACCENT, activeforeground="white",
                    font=("Menlo", 11),
                )
                om.grid(row=row, column=col * 3 + 1,
                        sticky="w", padx=(0, 10), pady=7)
                self._pwids[pid] = var

            elif ptype == "check":
                default = p.get("default", False)
                var = tk.BooleanVar(value=default)

                def _make_toggle(bvar, parent):
                    """Return a Button that flips bvar and recolours itself."""
                    btn = tk.Button(parent, relief="flat", cursor="hand2",
                                    bd=0, padx=10, pady=3,
                                    font=("Helvetica Neue", 11))

                    def _refresh():
                        if bvar.get():
                            btn.configure(text="● ON",
                                          bg=ACCENT, fg="white",
                                          activebackground="#6fa8ff",
                                          activeforeground="white")
                        else:
                            btn.configure(text="○ OFF",
                                          bg=BORDER, fg=MUTED,
                                          activebackground=CARD,
                                          activeforeground=MUTED)

                    def _toggle():
                        bvar.set(not bvar.get())
                        _refresh()

                    btn.configure(command=_toggle)
                    _refresh()          # set initial appearance
                    return btn

                btn = _make_toggle(var, self._pcard)
                btn.grid(row=row, column=col * 3 + 1,
                         sticky="w", padx=(0, 10), pady=7)
                self._pwids[pid] = var

            elif ptype == "file":
                var = tk.StringVar(value=p.get("default", ""))
                ff = tk.Frame(self._pcard, bg=CARD)
                ff.grid(row=row, column=col * 3 + 1, columnspan=4,
                        sticky="ew", padx=(0, 16), pady=7)
                tk.Entry(
                    ff, textvariable=var,
                    font=("Menlo", 10),
                    bg="#0d1117", fg=TEXT, insertbackground=TEXT,
                    relief="flat", bd=0, width=38,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                ).pack(side="left", padx=(0, 8))
                tk.Button(
                    ff, text="Browse…",
                    font=("Helvetica Neue", 10),
                    bg=BORDER, fg=TEXT, relief="flat", cursor="hand2",
                    command=lambda v=var: self._browse(v),
                ).pack(side="left")
                self._pwids[pid] = var
                col = (col + 1) % 2   # file spans wide — skip next column slot

            col = (col + 1) % 2
            if col == 0:
                row += 1

        # Bottom padding inside card
        tk.Frame(self._pcard, bg=CARD, height=10).grid(
            row=row + 2, column=0)
        self._pcard.columnconfigure(1, weight=1)
        self._pcard.columnconfigure(4, weight=1)

    # ── Command builder ──────────────────────────────────────────────────────────
    def _cmd(self):
        info = SCRIPTS[self._cur]
        cmd  = [sys.executable, info["path"]]

        for p in info["params"]:
            pid   = p["id"]
            ptype = p["type"]
            flag  = p.get("flag")
            var   = self._pwids.get(pid)
            if var is None:
                continue
            val = var.get()

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
                    cmd.append(val)   # positional argument

        return cmd

    # ── Button state helper ──────────────────────────────────────────────────────
    def _set_running(self, running: bool):
        self._running = running
        self._run_b.configure(state="disabled" if running else "normal")
        self._stp_b.configure(state="normal"   if running else "disabled")

    # ── Run / Stop ───────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return   # already running

        info = SCRIPTS[self._cur]

        # Validate required fields
        for p in info["params"]:
            if p.get("required"):
                v = self._pwids.get(p["id"])
                if v and not str(v.get()).strip():
                    self._log(f"  ⚠  '{p['label']}' is required.\n", "warn")
                    return

        cmd = self._cmd()
        cwd = os.path.dirname(info["path"])

        self._clear()
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"{'─' * 54}\n", "dim")
        self._log(f"  {info['icon']}   {self._cur}   ·   {ts}\n", "hdr")
        display = " ".join(
            os.path.basename(c) if i < 2 else c
            for i, c in enumerate(cmd)
        )
        self._log(f"  {display}\n", "dim")
        self._log(f"{'─' * 54}\n\n", "dim")

        self._set_running(True)

        def _thread():
            try:
                self._proc = subprocess.Popen(
                    cmd, cwd=cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    # start_new_session prevents grandchild processes (e.g. the
                    # browser launched by webbrowser.open) from inheriting the
                    # stdout pipe, which would block readline() indefinitely.
                    start_new_session=True,
                )
                for line in iter(self._proc.stdout.readline, ""):
                    self._q.put(line)
                self._proc.stdout.close()
                self._proc.wait()
                rc = self._proc.returncode
                self._q.put(
                    f"\n{'─' * 54}\n"
                    + (f"  ✓  Completed (exit 0)\n" if rc == 0
                       else f"  ✗  Exited with code {rc}\n")
                    + f"{'─' * 54}\n"
                )
            except Exception as exc:
                self._q.put(f"\nERROR: {exc}\n")
            finally:
                self._q.put(None)   # sentinel — tells poll() we're done

        threading.Thread(target=_thread, daemon=True).start()

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("\n  [Stopped by user]\n", "warn")
        self._set_running(False)

    # ── Output polling ───────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                if item is None:
                    # Sentinel: thread finished → re-enable Run button
                    self._set_running(False)
                else:
                    try:
                        self._log(item, self._tag(item))
                    except Exception:
                        pass  # never let a log error kill the poll loop
        except queue.Empty:
            pass
        except Exception:
            pass  # safety net — nothing should ever stop the poll loop
        finally:
            # Flush all buffered log lines in one widget pass (fast)
            self._flush_log()
            # Safety net: if the process has exited but the button is still
            # disabled (e.g. sentinel lost, thread crashed), re-enable it.
            if self._running and self._proc is not None:
                if self._proc.poll() is not None:
                    self._set_running(False)
            self.after(40, self._poll)

    def _tag(self, line: str) -> str:
        lo = line.lower()
        if any(x in lo for x in ("error", "traceback", "exception", "✗")):
            return "err"
        if any(x in lo for x in ("warning", "⚠", "skipped", "skip")):
            return "warn"
        if any(x in lo for x in ("✓", "saved →", "complete", " ok", "done")):
            return "ok"
        if line.startswith("  [") or line.startswith("["):
            return "info"
        if line.startswith(("╔", "║", "╚", "═")):
            return "hdr"
        return "ok"

    # ── Console helpers ──────────────────────────────────────────────────────────
    def _log(self, text: str, tag: str = "ok"):
        """Buffer a line — written to widget in one batch by _flush_log()."""
        self._log_buf.append((text, tag))

    def _flush_log(self):
        """Drain _log_buf into the ScrolledText in a single state-toggle pass."""
        if not self._log_buf:
            return
        self._con.configure(state="normal")
        for text, tag in self._log_buf:
            self._con.insert("end", text, tag)
        self._con.see("end")
        self._con.configure(state="disabled")
        self._log_buf.clear()

    def _clear(self):
        self._con.configure(state="normal")
        self._con.delete("1.0", "end")
        self._con.configure(state="disabled")

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Select portfolio CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=ROOT,
        )
        if path:
            var.set(path)


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
