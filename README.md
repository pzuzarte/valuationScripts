# ValuationSuite

A stock valuation and portfolio analysis toolkit. Provides multiple valuation methodologies, stock screeners, and portfolio dashboards — all output as styled HTML reports that open automatically in your browser.

---

## Installation

### Requirements

- **Python 3.9 or later** — [python.org/downloads](https://www.python.org/downloads/)
- **git** — [git-scm.com/downloads](https://git-scm.com/downloads)

No conda, no virtual environment setup, no admin rights needed. The installer handles everything.

---

### macOS / Linux

Paste this single command into your terminal:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.sh)
```

The installer will:
1. Clone the repository to `~/valuationScripts`
2. Create an isolated Python environment (`.venv`) inside the project
3. Install all dependencies
4. Create launchers — `ValuationSuite.app` (macOS), a `valuation-suite` terminal command, and `start.sh`

---

### Windows

Open **PowerShell** and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
iex (irm 'https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.ps1')
```

The installer will:
1. Clone the repository to `%USERPROFILE%\valuationScripts`
2. Create an isolated Python environment (`.venv`) inside the project
3. Install all dependencies
4. Create `start.bat` and a Desktop shortcut

---

### Install to a custom directory

```bash
# macOS / Linux
bash <(curl -fsSL https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.sh) --dir ~/mytools/valuation

# Windows PowerShell
.\install.ps1 -Dir C:\mytools\valuation
```

---

### Already cloned the repo?

Run the installer from inside the project folder — it will detect the existing clone automatically:

```bash
bash install.sh
```

---

## Launching

| Platform | Method |
|----------|--------|
| macOS | Double-click **ValuationSuite.app** |
| macOS / Linux | Run `valuation-suite` in a new terminal |
| macOS / Linux | Run `bash ~/valuationScripts/start.sh` |
| Windows | Double-click **ValuationSuite** on the Desktop |
| Windows | Run `start.bat` in the project folder |

The app opens at **http://127.0.0.1:5050** in your browser.

---

## Updating

Re-run the installer at any time — it pulls the latest code from GitHub and upgrades all packages:

```bash
bash install.sh
```

---

## Tools

| Tool | Description |
|------|-------------|
| **Value Screener** | Screens S&P 500, Nasdaq 100, Russell 2000, or TSX for undervalued stocks using DCF, P/E, P/FCF, EV/EBITDA, and PEG |
| **Growth Screener** | Four-pillar scoring model (Quality, Growth Momentum, Technical Signal, Valuation) |
| **Valuation Master** | Deep single-stock analysis — DCF, Reverse DCF, PEG, Monte Carlo, and more — with optional backtesting |
| **Portfolio Analyzer** | Multi-stock dashboard with signals, technicals, fundamentals, and rebalancing guidance |
| **Run Model** | Quick single-model valuation with optional backtest chart |

---

## Troubleshooting

**Python not found**
The installer searches for `python3.12`, `python3.11`, `python3.10`, `python3.9`, `python3`, and `python` in order. If none are found, install Python from [python.org/downloads](https://www.python.org/downloads/) and re-run the installer.

**`venv` module missing (Ubuntu/Debian)**
```bash
sudo apt install python3-venv
```

**Moved the project folder?**
Re-run `bash install.sh` — it rewrites all launchers with the new paths.

**Something broke after an update?**
The previous working version is on GitHub. You can roll back with:
```bash
cd ~/valuationScripts
git log --oneline -10        # find the commit you want
git checkout <commit-hash>
bash install.sh              # rebuild the venv against that version
```
