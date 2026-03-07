# ──────────────────────────────────────────────────────────────────────────────
#  ValuationSuite — Windows Installer (PowerShell)
#
#  Works on any Windows machine with Python 3.9+ and git installed.
#  Does NOT require conda, any named environment, or admin rights.
#
#  One-command install (no git clone needed first):
#    Set-ExecutionPolicy -Scope Process Bypass; iex (irm 'https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.ps1')
#
#  Or, after cloning the repo manually:
#    Set-ExecutionPolicy -Scope Process Bypass; .\install.ps1
#
#  Options:
#    -Dir C:\path\to\dir   Install to a custom directory (default: ~\valuationScripts)
#    -Help                 Show this help text
#
#  What this does:
#    1. Clones the repository — or detects that it's already present
#    2. Finds Python 3.9+ (py launcher, PATH, or common install locations)
#    3. Creates an isolated .venv inside the project
#    4. Installs all dependencies from requirements.txt into the venv
#    5. Creates launchers:
#         • start.bat            — run from the project directory
#         • ValuationSuite.lnk  — Desktop shortcut for double-click launch
# ──────────────────────────────────────────────────────────────────────────────

param(
    [string]$Dir  = "",
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────────────────
$REPO_URL          = "https://github.com/pzuzarte/valuationScripts.git"
$DEFAULT_INSTALL   = Join-Path $HOME "valuationScripts"
$VENV_DIR_NAME     = ".venv"
$MIN_PYTHON        = [Version]"3.9"
$TOTAL             = 5

# ── Helpers ────────────────────────────────────────────────────────────────────
function Step($n, $msg) { Write-Host "`n[$n/$TOTAL] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    [OK]  $msg"    -ForegroundColor Green }
function Warn($msg)     { Write-Host "    [!!]  $msg"    -ForegroundColor Yellow }
function Info($msg)     { Write-Host "    [..]  $msg"    -ForegroundColor Cyan }
function Fail($msg)     { Write-Host "`n[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ── Help ───────────────────────────────────────────────────────────────────────
if ($Help) {
    Write-Host @"
Usage:
  .\install.ps1 [-Dir C:\path\to\install]

One-command (no prior clone):
  Set-ExecutionPolicy -Scope Process Bypass
  iex (irm 'https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.ps1')
"@
    exit 0
}

# ── Banner ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ValuationSuite - Installer" -ForegroundColor Cyan
Write-Host "  ──────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "  Stock valuation and portfolio analysis toolkit"
Write-Host ""

# ── Step 1: Locate or clone the repository ─────────────────────────────────────
Step 1 "Repository"

$InstallDir  = if ($Dir) { $Dir } else { $DEFAULT_INSTALL }
$ProjectRoot = $null

# If PSScriptRoot is set and app.py is alongside this script, we're in the repo.
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "app.py"))) {
    $ProjectRoot = $PSScriptRoot
    Ok "Running from existing repository: $ProjectRoot"
} else {
    $ProjectRoot = $InstallDir

    if (Test-Path (Join-Path $ProjectRoot ".git")) {
        Info "Repository already exists at $ProjectRoot - pulling latest changes"
        try {
            $pullOut = git -C $ProjectRoot pull --ff-only 2>&1
            ($pullOut | Select-Object -Last 1) | ForEach-Object { Write-Host "    $_" }
            Ok "Repository up to date"
        } catch {
            Warn "Could not fast-forward - local changes may be present."
            Warn "If you want the latest code, commit or stash your changes and re-run."
        }
    } else {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Fail "git is required but not installed.`nDownload from: https://git-scm.com/downloads"
        }
        Info "Cloning into $ProjectRoot ..."
        git clone --depth=1 $REPO_URL $ProjectRoot
        Ok "Cloned $REPO_URL"
    }
}

if (-not (Test-Path (Join-Path $ProjectRoot "app.py"))) {
    Fail "app.py not found in $ProjectRoot - the clone may have failed."
}

Set-Location $ProjectRoot

# ── Step 2: Find Python 3.9+ ──────────────────────────────────────────────────
Step 2 "Python 3.9+"

# Candidate commands in priority order.
# 'py -3.X' is the Windows Python Launcher (most reliable on Windows).
$PythonCandidates = @(
    "py -3.13", "py -3.12", "py -3.11", "py -3.10", "py -3.9",
    "python3.13", "python3.12", "python3.11", "python3.10", "python3.9",
    "python3", "python"
)

$PythonCmd = $null
$PythonVer = $null

foreach ($candidate in $PythonCandidates) {
    $parts = $candidate -split " ", 2
    $exe   = $parts[0]
    $arg   = if ($parts.Count -gt 1) { $parts[1] } else { $null }

    try {
        $verArgs = @("-c", "import sys; print(sys.version_info.major, sys.version_info.minor)")
        if ($arg) { $verArgs = @($arg) + $verArgs }

        $verOut = & $exe @verArgs 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $verOut) { continue }

        $nums = $verOut.Trim() -split "\s+"
        $v    = [Version]("$($nums[0]).$($nums[1])")

        if ($v -ge $MIN_PYTHON) {
            # Build the invoke expression we'll reuse later
            $PythonCmd = if ($arg) { "$exe $arg" } else { $exe }
            $verDisplay = & $exe ($arg ? @($arg, "--version") : @("--version")) 2>&1
            $PythonVer  = "$verDisplay"
            break
        }
    } catch { }
}

if (-not $PythonCmd) {
    Fail "Python 3.9+ not found.`n`nInstall from: https://www.python.org/downloads/`n(Tick 'Add Python to PATH' during installation)"
}

Ok "$PythonVer"

# Helper to invoke the chosen Python (handles 'py -3.11 ...' form).
function Invoke-Python {
    param([string[]]$Arguments)
    $parts = $PythonCmd -split " ", 2
    $exe   = $parts[0]
    $pre   = if ($parts.Count -gt 1) { @($parts[1]) } else { @() }
    & $exe @pre @Arguments
}

# Check venv module is available.
$venvCheck = Invoke-Python @("-m", "venv", "--help") 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail "The 'venv' module is not available in the found Python.`nInstall a full Python distribution from https://www.python.org/downloads/"
}

# ── Step 3: Virtual environment ───────────────────────────────────────────────
Step 3 "Virtual environment"

$VenvPath   = Join-Path $ProjectRoot $VENV_DIR_NAME
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip    = Join-Path $VenvPath "Scripts\pip.exe"

if (Test-Path $VenvPython) {
    Ok "Existing venv found at $VenvPath"
} else {
    Info "Creating $VenvPath ..."
    Invoke-Python @("-m", "venv", $VenvPath)
    Ok "Virtual environment created"
}

Info "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip --quiet

# ── Step 4: Install packages ──────────────────────────────────────────────────
Step 4 "Installing packages"

$ReqFile = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $ReqFile)) { Fail "requirements.txt not found at $ReqFile" }

Info "pip install -r requirements.txt  (first run takes ~2-3 minutes) ..."
& $VenvPython -m pip install -r $ReqFile --upgrade --quiet 2>&1 `
    | Where-Object { $_ -notmatch "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING:|Using cached)" } `
    | Where-Object { $_.Trim() -ne "" } `
    | ForEach-Object { Write-Host "    $_" }

# Spot-check critical imports.
$Missing = @()
foreach ($pkg in @("flask", "yfinance", "tradingview_screener", "pandas", "numpy", "scipy", "matplotlib")) {
    & $VenvPython -c "import $pkg" 2>$null
    if ($LASTEXITCODE -ne 0) { $Missing += $pkg }
}

if ($Missing.Count -gt 0) {
    Warn "Could not import: $($Missing -join ', ')"
    Warn "Try re-running this script, or: $VenvPip install $($Missing -join ' ')"
} else {
    Ok "All packages verified"
}

# ── Step 5: Launchers ─────────────────────────────────────────────────────────
Step 5 "Creating launchers"

# Ensure expected output directories exist.
foreach ($subdir in @("valueData", "growthData", "valuationData", "portfolioData", "plots",
                      "8_watchlist\watchlistData")) {
    $p = Join-Path $ProjectRoot $subdir
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}
Ok "Output directories verified"

# ── 5a. start.bat ─────────────────────────────────────────────────────────────
$StartBat = Join-Path $ProjectRoot "start.bat"
$startContent = @"
@echo off
REM ValuationSuite - project launcher
REM Generated by install.ps1 on $(Get-Date -Format "yyyy-MM-dd").
REM Re-run install.ps1 if you move the project directory.
"$VenvPython" "$ProjectRoot\app.py" %*
"@
Set-Content -Path $StartBat -Value $startContent -Encoding UTF8
Ok "start.bat written"

# ── 5b. Desktop shortcut ──────────────────────────────────────────────────────
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "ValuationSuite.lnk"

try {
    $WshShell          = New-Object -ComObject WScript.Shell
    $Shortcut          = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath       = $VenvPython
    $Shortcut.Arguments        = "`"$ProjectRoot\app.py`""
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.Description      = "ValuationSuite - Stock Valuation Toolkit"
    $Shortcut.WindowStyle      = 7   # minimised — browser is the real UI
    $Shortcut.Save()
    Ok "Desktop shortcut: ValuationSuite.lnk"
} catch {
    Warn "Could not create desktop shortcut: $_"
    Warn "(Non-critical — use start.bat to launch)"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Project location : $ProjectRoot"
Write-Host "  Python           : $VenvPython"
Write-Host ""
Write-Host "  How to launch:"
Write-Host "  [1]  Double-click   ValuationSuite.lnk  on your Desktop"
Write-Host "  [2]  Command line:  $StartBat"
Write-Host "  [3]  PowerShell:    & `"$VenvPython`" `"$ProjectRoot\app.py`""
Write-Host ""
Write-Host "  Updating later:      re-run install.ps1 - pulls latest code, upgrades packages."
Write-Host "  Moving the project:  re-run install.ps1 to rebake launcher paths."
Write-Host ""
