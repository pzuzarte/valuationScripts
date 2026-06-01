#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  ValuationSuite — Universal Installer
#
#  Works on any macOS or Linux machine with Python 3.11+ and git installed.
#  Does NOT require conda, py311, or any pre-existing environment.
#
#  One-command install (no git clone needed first):
#    bash <(curl -fsSL https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.sh)
#
#  Or, after cloning the repo manually:
#    bash install.sh
#
#  Options:
#    --dir /path/to/dir   Install to a custom directory (default: ~/valuationScripts)
#    --help               Show this help text
#
#  What this does:
#    1. Checks prerequisites (git, Python 3.9+, Xcode CLT on macOS)
#    2. Clones the repository — or detects that it's already present
#    3. Creates an isolated .venv inside the project
#    4. Installs all dependencies from requirements.txt into the venv
#    5. Installs PyTorch (CPU wheel — works on macOS + Linux without CUDA)
#    6. Creates launchers:
#         • start.sh           — run from the project directory
#         • valuation-suite    — global CLI command in ~/bin
#         • ValuationSuite.app — macOS double-click launcher (macOS only)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/pzuzarte/valuationScripts.git"
DEFAULT_INSTALL_DIR="$HOME/valuationScripts"
VENV_DIR_NAME=".venv"
MIN_PYTHON_MINOR=11         # require Python 3.11+ (3.9/3.10 are EOL or near-EOL;
                            # 3.11 is needed for TimesFM and full package compatibility)
TOTAL=6

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'; BLD='\033[1m'

step()  { echo -e "\n${BLU}[$1/$TOTAL]${RST} ${BLD}$2${RST}"; }
ok()    { echo -e "    ${GRN}✓${RST}  $1"; }
warn()  { echo -e "    ${YLW}⚠${RST}  $1"; }
info()  { echo -e "    ${CYN}•${RST}  $1"; }
fail()  { echo -e "\n${RED}✗ Error:${RST} $1"; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
INSTALL_DIR="$DEFAULT_INSTALL_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)   shift; INSTALL_DIR="$1" ;;
        --dir=*) INSTALL_DIR="${1#--dir=}" ;;
        --help|-h)
            echo -e "Usage: bash install.sh [--dir /path/to/install]"
            echo -e "       bash <(curl -fsSL https://raw.githubusercontent.com/pzuzarte/valuationScripts/main/install.sh) [--dir /path]"
            exit 0 ;;
        *) warn "Unknown option: $1 (ignored)" ;;
    esac
    shift
done

# ── Banner ─────────────────────────────────────────────────────────────────────
echo -e "\n${BLD}${CYN}  ValuationSuite — Installer${RST}"
echo -e "  ${CYN}────────────────────────────────────────${RST}"
echo -e "  Stock valuation and portfolio analysis toolkit\n"

IS_MACOS=false
[[ "$(uname -s)" == "Darwin" ]] && IS_MACOS=true

# ── Step 1: Prerequisites ─────────────────────────────────────────────────────
step 1 "Prerequisites"

# macOS: check for Xcode Command Line Tools (needed to compile C extensions
# such as hdbscan, wordcloud, dtaidistance, etc.)
if $IS_MACOS; then
    if ! xcode-select -p >/dev/null 2>&1; then
        warn "Xcode Command Line Tools are not installed."
        warn "Some packages (hdbscan, wordcloud, etc.) require a C compiler."
        echo ""
        echo -e "  ${BLD}Installing Xcode Command Line Tools now…${RST}"
        echo -e "  A dialog will appear — click ${BLD}Install${RST} and wait for it to finish."
        echo -e "  Then re-run this script.\n"
        xcode-select --install 2>/dev/null || true
        echo -e "\n${YLW}Re-run install.sh after the Xcode tools finish installing.${RST}"
        exit 0
    else
        ok "Xcode Command Line Tools: $(xcode-select -p)"
    fi

    # libomp — required by xgboost on macOS (OpenMP runtime).
    # brew is the only practical way to get it; skip silently if brew absent.
    if command -v brew >/dev/null 2>&1; then
        if ! brew list libomp &>/dev/null; then
            info "Installing libomp (required by xgboost) …"
            brew install libomp --quiet 2>&1 | tail -1 | sed 's/^/    /' || true
            brew list libomp &>/dev/null && ok "libomp installed" || \
                warn "libomp install failed — XGBoost may not load (run: brew install libomp)"
        else
            ok "libomp already installed"
        fi
    else
        # No brew — check if libomp is already on the system anyway
        if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib && \
              ! -f /usr/local/opt/libomp/lib/libomp.dylib ]]; then
            warn "Homebrew not found and libomp is missing — XGBoost may not load."
            warn "Fix: install Homebrew (https://brew.sh) then run: brew install libomp"
        else
            ok "libomp found"
        fi
    fi
fi

# Linux: ensure libgomp (GNU OpenMP) is present for xgboost.
# The KMP_DUPLICATE_LIB_OK env var handles the runtime conflict with PyTorch's
# bundled OpenMP, but the library still needs to be installed.
if ! $IS_MACOS; then
    if command -v apt-get >/dev/null 2>&1; then
        if ! ldconfig -p 2>/dev/null | grep -q libgomp; then
            info "Installing libgomp (required by xgboost on Linux) …"
            sudo apt-get install -y libgomp1 --quiet 2>/dev/null || \
                warn "Could not install libgomp — XGBoost may not load (run: sudo apt-get install libgomp1)"
        else
            ok "libgomp found"
        fi
    elif command -v yum >/dev/null 2>&1; then
        rpm -q libgomp &>/dev/null || sudo yum install -y libgomp --quiet 2>/dev/null || true
    fi
fi

# git
command -v git >/dev/null 2>&1 || fail \
    "git is required but not installed.\n\
  macOS:  xcode-select --install   (installs Xcode Command Line Tools)\n\
  Ubuntu: sudo apt install git\n\
  Other:  https://git-scm.com/downloads"
ok "git: $(git --version)"

# ── Step 2: Locate or clone the repository ─────────────────────────────────────
step 2 "Repository"

# Detect whether we are running from inside an already-cloned repo.
#
# Three invocation styles and what BASH_SOURCE[0] contains:
#   bash install.sh              → absolute or relative path to the script   ✓
#   bash <(curl -fsSL ...)       → /dev/fd/63  (process substitution fd)     ✗
#   curl ... | bash              → empty string                               ✗
#
# Guard: skip the "inside repo" branch whenever the source path starts with
# /dev/ (process substitution) or is empty.
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
PROJECT_ROOT=""

if [[ -n "$SCRIPT_SOURCE" && "$SCRIPT_SOURCE" != /dev/* && -f "$SCRIPT_SOURCE" ]]; then
    _candidate="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    if [[ -f "$_candidate/app.py" ]]; then
        PROJECT_ROOT="$_candidate"
        ok "Running from existing repository: $PROJECT_ROOT"
    fi
fi

if [[ -z "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT="$INSTALL_DIR"

    if [[ -d "$PROJECT_ROOT/.git" ]]; then
        info "Repository already exists at $PROJECT_ROOT — pulling latest changes"
        _pull_out="$(git -C "$PROJECT_ROOT" pull --ff-only 2>&1)" || {
            warn "Could not fast-forward — local changes may be present."
            warn "If you want the latest code, commit or stash your changes and re-run."
        }
        echo "$_pull_out" | tail -1 | sed 's/^/    /'
        ok "Repository up to date"
        # Clear Python bytecode cache so updated .py files are always used.
        find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
        ok "Bytecode cache cleared"
    else
        info "Cloning into $PROJECT_ROOT …"
        git clone --depth=1 "$REPO_URL" "$PROJECT_ROOT"
        ok "Cloned $REPO_URL"
    fi
fi

[[ -f "$PROJECT_ROOT/app.py" ]] || fail "app.py not found in $PROJECT_ROOT — the clone may have failed."

cd "$PROJECT_ROOT"

# ── Step 3: Find Python 3.9+ ──────────────────────────────────────────────────
step 3 "Python 3.9+"

# Try specific version binaries first (most precise), then generic names.
PYTHON_CANDIDATES=(
    "python3.13" "python3.12" "python3.11" "python3.10" "python3.9"
    "python3" "python"
)
PYTHON_CMD=""

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
        _ver="$("$candidate" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null)" || continue
        _maj="${_ver%% *}"
        _min="${_ver##* }"
        if [[ "$_maj" -ge 3 && "$_min" -ge "$MIN_PYTHON_MINOR" ]] 2>/dev/null; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    fail "Python 3.${MIN_PYTHON_MINOR}+ is required but was not found.\n\
\n\
  Install options:\n\
    macOS:  brew install python@3.11\n\
            — or — https://www.python.org/downloads/\n\
    Ubuntu: sudo apt install python3.11 python3.11-venv\n\
    Fedora: sudo dnf install python3.11\n\
    Any:    https://www.python.org/downloads/\n\
\n\
  After installing, re-run:  bash install.sh"
fi

PYTHON_VER="$("$PYTHON_CMD" --version 2>&1)"
ok "$PYTHON_VER  ($(command -v "$PYTHON_CMD"))"

# Some distros (Ubuntu/Debian) package venv separately (python3-venv).
"$PYTHON_CMD" -m venv --help >/dev/null 2>&1 || fail \
    "The 'venv' module is missing from $PYTHON_CMD.\n\
  Ubuntu/Debian: sudo apt install python3-venv\n\
  Fedora:        sudo dnf install python3-venv"

# ── Step 4: Virtual environment ───────────────────────────────────────────────
step 4 "Virtual environment"

VENV_PATH="$PROJECT_ROOT/$VENV_DIR_NAME"

if [[ -d "$VENV_PATH/bin" ]]; then
    _venv_minor="$("$VENV_PATH/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")"
    if [[ "$_venv_minor" -ge "$MIN_PYTHON_MINOR" ]]; then
        ok "Existing venv found at $VENV_PATH (Python 3.${_venv_minor})"
    else
        info "Existing venv uses Python 3.${_venv_minor} — too old, rebuilding with $PYTHON_CMD …"
        rm -rf "$VENV_PATH"
        "$PYTHON_CMD" -m venv "$VENV_PATH"
        ok "Virtual environment rebuilt ($(${VENV_PATH}/bin/python --version))"
    fi
else
    info "Creating $VENV_PATH …"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
    ok "Virtual environment created ($(${VENV_PATH}/bin/python --version))"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

info "Upgrading pip …"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# ── Step 5: Install packages ──────────────────────────────────────────────────
step 5 "Installing packages"

REQ_FILE="$PROJECT_ROOT/requirements.txt"
[[ -f "$REQ_FILE" ]] || fail "requirements.txt not found at $REQ_FILE"

info "pip install -r requirements.txt  (first run takes ~2-5 minutes) …"
# --prefer-binary: use pre-built wheels instead of compiling from source.
# This avoids C-compiler failures for packages like hdbscan, wordcloud, etc.
# on machines where Xcode CLT was just installed or pip wheels are available.
# NOTE: pip may report a chronos-forecasting / transformers version conflict.
# This is a false alarm -- chronos 2.x works correctly with transformers 5.x;
# the package metadata simply hasn't been updated to reflect this.
"$VENV_PYTHON" -m pip install -r "$REQ_FILE" \
    --prefer-binary \
    --upgrade \
    --quiet 2>&1 \
    | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip|Using cached)" \
    | grep -vE "chronos-forecasting|pip.s dependency resolver|source of the following dependency" \
    | grep -v "^$" \
    || true

# Some packages (hdbscan, wordcloud, tslearn) occasionally fail even with
# --prefer-binary on unusual Python versions.  Try each separately so that
# one failure doesn't block everything else.
COMPILE_PKGS=("hdbscan>=0.8.33" "wordcloud>=1.9.0" "tslearn>=0.6.0" "dtaidistance>=2.3.0")
for pkg in "${COMPILE_PKGS[@]}"; do
    pkg_name="${pkg%%[>=<]*}"
    if ! "$VENV_PYTHON" -c "import ${pkg_name//-/_}" 2>/dev/null; then
        info "Re-trying $pkg_name with binary-only install …"
        "$VENV_PYTHON" -m pip install "$pkg" \
            --prefer-binary --only-binary=:all: \
            --quiet 2>&1 | grep -v "^$" || \
        warn "$pkg_name could not be installed — classifier clustering features may be limited"
    fi
done

# Prophet: has a C compiler dependency (pystan) — retry separately so a compile
# failure doesn't block the rest of the install.
if ! "$VENV_PYTHON" -c "from prophet import Prophet" 2>/dev/null; then
    info "Installing Prophet (additive forecasting model) …"
    "$VENV_PYTHON" -m pip install prophet \
        --prefer-binary \
        --quiet 2>&1 | grep -v "^$" || \
    warn "Prophet could not be installed — Prophet forecasting will be unavailable"
fi

# ── PyTorch (CPU wheel) ───────────────────────────────────────────────────────
# torch is NOT in requirements.txt — we install it here from the official CPU
# wheel index so the correct binary is always used (avoids the 2 GB CUDA
# download on Linux, and avoids a double-install on macOS).
if ! "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
    info "Installing PyTorch CPU wheel (~750 MB, one-time download) …"
    "$VENV_PYTHON" -m pip install torch \
        --index-url https://download.pytorch.org/whl/cpu \
        --quiet 2>&1 \
        | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip|Using cached)" \
        | grep -v "^$" \
        || true
    if "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
        ok "PyTorch installed"
    else
        warn "PyTorch install failed — FinBERT sentiment and LSTM forecasting will be unavailable"
    fi
else
    ok "PyTorch already present"
fi

# ── Foundation time-series models (optional, ~200-300 MB each) ───────────────
# TimesFM (Google) and Chronos (Amazon) are zero-shot forecasting models used
# by priceForecast.  They are large and have complex dependencies, so we install
# them only if they are not already present, and failures are non-fatal.
#
# TimesFM requires Python ≥3.10 and the torch backend (installed above).
# Chronos requires Python ≥3.9 and torch.
if ! "$VENV_PYTHON" -c "import timesfm" 2>/dev/null; then
    info "Installing TimesFM (Google zero-shot forecasting) …"
    # Redirect both stdout and stderr — timesfm has unstable transitive deps
    # (paxml/praxis) that produce noisy ERROR lines even when pip falls back
    # gracefully.  We check success via import, not pip's exit code.
    "$VENV_PYTHON" -m pip install timesfm \
        --prefer-binary \
        --quiet 2>/dev/null || true
    if "$VENV_PYTHON" -c "import timesfm" 2>/dev/null; then
        ok "TimesFM installed"
    else
        warn "TimesFM install failed — TimesFM forecasting will be unavailable (pip install timesfm to retry)"
    fi
else
    ok "TimesFM already present"
fi

if ! "$VENV_PYTHON" -c "from chronos import BaseChronosPipeline" 2>/dev/null; then
    info "Installing Chronos-Bolt (Amazon zero-shot forecasting) …"
    "$VENV_PYTHON" -m pip install chronos-forecasting \
        --prefer-binary \
        --quiet 2>&1 \
        | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip|Using cached)" \
        | grep -v "^$" \
        || true
    if "$VENV_PYTHON" -c "from chronos import BaseChronosPipeline" 2>/dev/null; then
        ok "Chronos installed"
    else
        warn "Chronos install failed — Chronos forecasting will be unavailable (pip install chronos-forecasting to retry)"
    fi
else
    ok "Chronos already present"
fi

# ── TFT — Temporal Fusion Transformer (optional, requires torch) ──────────────
# pytorch-forecasting + lightning are large (~300 MB combined). They require
# torch to already be installed (done above). Skipped silently if torch failed.
if "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
    if ! "$VENV_PYTHON" -c "from pytorch_forecasting import TemporalFusionTransformer" 2>/dev/null; then
        info "Installing pytorch-forecasting + lightning (TFT model) …"
        "$VENV_PYTHON" -m pip install pytorch-forecasting lightning \
            --prefer-binary 2>&1 \
            | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip|Using cached)" \
            | grep -v "^$" \
            || true
        if "$VENV_PYTHON" -c "from pytorch_forecasting import TemporalFusionTransformer" 2>/dev/null; then
            ok "pytorch-forecasting + lightning installed (TFT enabled)"
        else
            warn "pytorch-forecasting install failed — TFT forecasting will be unavailable (pip install pytorch-forecasting lightning to retry)"
        fi
    else
        ok "pytorch-forecasting already present"
    fi
else
    warn "torch not available — skipping pytorch-forecasting / TFT install"
fi

# N-HiTS and N-BEATS are implemented natively in PyTorch (priceForecast.py) —
# no neuralforecast package needed.  neuralforecast is intentionally NOT installed
# here because it has a pyarrow/protobuf binary conflict with conda environments.

# Spot-check critical imports.
CRITICAL=(flask yfinance pandas numpy scipy matplotlib requests feedparser vaderSentiment transformers xgboost arch sklearn)
MISSING=()
for pkg in "${CRITICAL[@]}"; do
    "$VENV_PYTHON" -c "import ${pkg//-/_}" 2>/dev/null || MISSING+=("$pkg")
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Could not import: ${MISSING[*]}"
    warn "Try re-running this script, or: $VENV_PIP install ${MISSING[*]}"
else
    ok "All critical packages verified"
fi

# Deep-check: actually run a tiny forecast to confirm torch + xgboost load cleanly
# together (the OpenMP conflict only surfaces at runtime, not at import).
info "Running quick smoke-test (torch + xgboost load check) …"
SMOKE_OK=false
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    "$VENV_PYTHON" - <<'PYSMOKE' 2>/dev/null && SMOKE_OK=true
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from xgboost import XGBRegressor
import torch, numpy as np
x = torch.tensor([[1.0, 2.0]])
_ = torch.nn.Linear(2, 1)(x)
print("ok")
PYSMOKE
$SMOKE_OK && ok "torch + xgboost load cleanly — no OpenMP conflict" || \
    warn "Smoke-test failed — torch or xgboost may have a runtime conflict on this machine"

# ── Step 6: Launchers + project setup ─────────────────────────────────────────
step 6 "Project setup & launchers"

# ── 6a. Output directories ────────────────────────────────────────────────────
# Scripts create these at runtime via os.makedirs(exist_ok=True), but creating
# them here ensures a clean first run without any directory-not-found errors.
OUTPUT_DIRS=(
    # root-level (legacy scripts write here)
    valueData growthData valuationData portfolioData scatterData plots
    # per-script output dirs
    5_scatterPlots/scatterData
    6_sentimentAnalyzer/sentimentData
    7_macroDashboard/macroData
    8_watchlist/watchlistData
    9_researchScanner/researchData
    10_classifier/classifierData
    11_growthScreeners/output
    11_priceForecast/forecastData
    12_magicFormula/formulaData
    13_qualityScreener/qualityData
    14_canslim/canslimData
    15_earningsAccel/accelData
    16_topicSentiment/topicData
    # predictive model output dirs
    17_returnPredictor/predictorData
    18_earningsPredictor/earningsPredData
    19_regimeDetector/regimeData
    20_riskModel/riskData
    21_breakoutDetector/breakoutData
    22_sentimentLead/sentimentLeadData
    deepDiveTickers
)
for dir in "${OUTPUT_DIRS[@]}"; do
    mkdir -p "$PROJECT_ROOT/$dir"
done
ok "Output directories verified (${#OUTPUT_DIRS[@]} dirs)"

# ── 6b. Default CSV stubs ─────────────────────────────────────────────────────
# Create empty-but-valid CSV stubs for files that scripts expect to exist.
# Only written if the file does not already exist so user data is preserved.
_write_csv_stub() {
    local path="$1" header="$2"
    if [[ ! -f "$path" ]]; then
        echo "$header" > "$path"
        info "Created default: $path"
    fi
}

_write_csv_stub "$PROJECT_ROOT/deepDiveTickers/deepDiveTickers.csv"     "ticker,shares,added_date"
_write_csv_stub "$PROJECT_ROOT/4_portfolioAnalyzer/deepDiveTickers.csv" "ticker,shares,added_date"
_write_csv_stub "$PROJECT_ROOT/4_portfolioAnalyzer/currentlyOwnded.csv" "ticker,shares"
ok "Default CSV stubs ready"

# ── 6c. start.sh ──────────────────────────────────────────────────────────────
START_SH="$PROJECT_ROOT/start.sh"
cat > "$START_SH" <<STARTSH
#!/usr/bin/env bash
# ValuationSuite — project launcher
# Generated by install.sh on $(date "+%Y-%m-%d").
# Re-run install.sh if you move the project directory.
# KMP_DUPLICATE_LIB_OK: prevents SIGSEGV when brew's libomp and PyTorch's
# bundled libiomp5 are both present on the same macOS machine.
export KMP_DUPLICATE_LIB_OK=TRUE
exec "${VENV_PYTHON}" "${PROJECT_ROOT}/app.py" "\$@"
STARTSH
chmod +x "$START_SH"
ok "start.sh written"

# ── 6d. ~/bin/valuation-suite ─────────────────────────────────────────────────
BIN_DIR="$HOME/bin"
mkdir -p "$BIN_DIR"
CLI_SCRIPT="$BIN_DIR/valuation-suite"
cat > "$CLI_SCRIPT" <<CLISCRIPT
#!/usr/bin/env bash
# valuation-suite — CLI launcher for ValuationSuite
# Generated by install.sh on $(date "+%Y-%m-%d").
# Re-run install.sh if you move the project directory.
export KMP_DUPLICATE_LIB_OK=TRUE
exec "${VENV_PYTHON}" "${PROJECT_ROOT}/app.py" "\$@"
CLISCRIPT
chmod +x "$CLI_SCRIPT"
ok "~/bin/valuation-suite written"

# Add ~/bin to PATH if it isn't already there.
SHELL_PROFILE=""
if [[ "${SHELL:-}" == *"zsh"* ]];  then SHELL_PROFILE="$HOME/.zshrc"; fi
if [[ "${SHELL:-}" == *"bash"* ]]; then SHELL_PROFILE="$HOME/.bash_profile"; fi

PATH_LINE='export PATH="$HOME/bin:$PATH"'

if [[ ":${PATH}:" == *":${HOME}/bin:"* ]]; then
    ok "~/bin already in PATH"
elif [[ -n "$SHELL_PROFILE" ]]; then
    grep -qxF "$PATH_LINE" "$SHELL_PROFILE" 2>/dev/null || echo "$PATH_LINE" >> "$SHELL_PROFILE"
    ok "Added ~/bin to PATH in $SHELL_PROFILE"
    info "Run:  source $SHELL_PROFILE   (or open a new terminal)"
else
    warn "Could not detect shell profile — add this line manually:"
    info "  $PATH_LINE"
fi

# ── 6e. macOS: rebuild ValuationSuite.app ────────────────────────────────────
if $IS_MACOS; then
    APP_DIR="$PROJECT_ROOT/ValuationSuite.app"
    MACOS_DIR="$APP_DIR/Contents/MacOS"
    RES_DIR="$APP_DIR/Contents/Resources"
    mkdir -p "$MACOS_DIR" "$RES_DIR"

    cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>         <string>ValuationSuite</string>
    <key>CFBundleIdentifier</key>         <string>com.valuationsuite.app</string>
    <key>CFBundleName</key>               <string>Valuation Suite</string>
    <key>CFBundleDisplayName</key>        <string>Valuation Suite</string>
    <key>CFBundleVersion</key>            <string>2.1</string>
    <key>CFBundleShortVersionString</key> <string>2.1</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>CFBundleSignature</key>          <string>????</string>
    <key>NSHighResolutionCapable</key>    <true/>
    <key>LSMinimumSystemVersion</key>     <string>10.13</string>
    <key>LSUIElement</key>                <false/>
    <key>CFBundleIconFile</key>           <string>AppIcon</string>
</dict>
</plist>
PLIST

    ICNS_SRC="$PROJECT_ROOT/AppIcon.icns"
    if [[ -f "$ICNS_SRC" ]]; then
        cp "$ICNS_SRC" "$RES_DIR/AppIcon.icns"
        touch "$APP_DIR"
        ok "App icon applied"
    else
        warn "AppIcon.icns not found — app will use default icon"
    fi

    cat > "$MACOS_DIR/ValuationSuite" <<LAUNCHER
#!/usr/bin/env bash
# ValuationSuite.app launcher — generated by install.sh on $(date "+%Y-%m-%d").
# Re-run install.sh if you move the project directory.

if [[ ! -f "${PROJECT_ROOT}/app.py" ]]; then
    osascript -e 'display alert "ValuationSuite — Project Not Found" message "The project was not found at its installed location.\n\nRe-run install.sh to rebuild this launcher." as critical' 2>/dev/null || true
    exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE
exec "${VENV_PYTHON}" "${PROJECT_ROOT}/app.py"
LAUNCHER
    chmod +x "$MACOS_DIR/ValuationSuite"
    ok "ValuationSuite.app rebuilt (${APP_DIR})"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}${BLD}✓ Installation complete!${RST}"
echo ""
echo -e "  ${BLD}Project location:${RST} ${CYN}${PROJECT_ROOT}${RST}"
echo -e "  ${BLD}Python:${RST}           ${CYN}${VENV_PYTHON}${RST}"
echo ""
echo -e "  ${BLD}How to launch:${RST}"
if $IS_MACOS; then
echo -e "  ${CYN}①${RST}  Double-click  ${BLD}ValuationSuite.app${RST}  (copy to Desktop or Dock)"
fi
echo -e "  ${CYN}②${RST}  Terminal:     ${BLD}valuation-suite${RST}  (after opening a new terminal tab)"
echo -e "  ${CYN}③${RST}  Terminal:     ${BLD}bash ${PROJECT_ROOT}/start.sh${RST}"
echo ""
echo -e "  ${BLD}Updating later:${RST}   re-run ${CYN}bash install.sh${RST} — pulls latest code, upgrades packages."
echo -e "  ${BLD}Moving the project:${RST} re-run ${CYN}bash install.sh${RST} to rebake launcher paths."
echo ""
