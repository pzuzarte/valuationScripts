#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  ValuationSuite — Universal Installer
#
#  Works on any macOS or Linux machine with Python 3.9+ and git installed.
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
#    1. Clones the repository — or detects that it's already present
#    2. Finds Python 3.9+ (system Python, Homebrew, pyenv — anything works)
#    3. Creates an isolated .venv inside the project
#    4. Installs all dependencies from requirements.txt into the venv
#    5. Creates launchers:
#         • start.sh           — run from the project directory
#         • valuation-suite    — global CLI command in ~/bin
#         • ValuationSuite.app — macOS double-click launcher (macOS only)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/pzuzarte/valuationScripts.git"
DEFAULT_INSTALL_DIR="$HOME/valuationScripts"
VENV_DIR_NAME=".venv"
MIN_PYTHON_MINOR=9          # require Python 3.9+
TOTAL=5

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

# ── Step 1: Locate or clone the repository ─────────────────────────────────────
step 1 "Repository"

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
        # Use || true so a dirty working tree doesn't abort the install
        _pull_out="$(git -C "$PROJECT_ROOT" pull --ff-only 2>&1)" || {
            warn "Could not fast-forward — local changes may be present."
            warn "If you want the latest code, commit or stash your changes and re-run."
        }
        echo "$_pull_out" | tail -1 | sed 's/^/    /'
        ok "Repository up to date"
    else
        command -v git >/dev/null 2>&1 || fail \
            "git is required but not installed.\n\
  macOS:  xcode-select --install   (installs Xcode Command Line Tools)\n\
  Ubuntu: sudo apt install git\n\
  Other:  https://git-scm.com/downloads"

        info "Cloning into $PROJECT_ROOT …"
        git clone --depth=1 "$REPO_URL" "$PROJECT_ROOT"
        ok "Cloned $REPO_URL"
    fi
fi

[[ -f "$PROJECT_ROOT/app.py" ]] || fail "app.py not found in $PROJECT_ROOT — the clone may have failed."

cd "$PROJECT_ROOT"

# ── Step 2: Find Python 3.9+ ──────────────────────────────────────────────────
step 2 "Python 3.9+"

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
    macOS:  brew install python@3.12\n\
            — or — https://www.python.org/downloads/\n\
    Ubuntu: sudo apt install python3.12 python3.12-venv\n\
    Fedora: sudo dnf install python3.12\n\
    Any:    https://www.python.org/downloads/"
fi

PYTHON_VER="$("$PYTHON_CMD" --version 2>&1)"
ok "$PYTHON_VER  ($(command -v "$PYTHON_CMD"))"

# Some distros (Ubuntu/Debian) package venv separately (python3-venv).
"$PYTHON_CMD" -m venv --help >/dev/null 2>&1 || fail \
    "The 'venv' module is missing from $PYTHON_CMD.\n\
  Ubuntu/Debian: sudo apt install python3-venv\n\
  Fedora:        sudo dnf install python3-venv"

# ── Step 3: Virtual environment ───────────────────────────────────────────────
step 3 "Virtual environment"

VENV_PATH="$PROJECT_ROOT/$VENV_DIR_NAME"

if [[ -d "$VENV_PATH/bin" ]]; then
    ok "Existing venv found at $VENV_PATH"
else
    info "Creating $VENV_PATH …"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
    ok "Virtual environment created"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

info "Upgrading pip …"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# ── Step 4: Install packages ──────────────────────────────────────────────────
step 4 "Installing packages"

REQ_FILE="$PROJECT_ROOT/requirements.txt"
[[ -f "$REQ_FILE" ]] || fail "requirements.txt not found at $REQ_FILE"

info "pip install -r requirements.txt  (first run takes ~2 minutes) …"
# Suppress the per-line install noise but show any real warnings/errors.
"$VENV_PYTHON" -m pip install -r "$REQ_FILE" --upgrade --quiet 2>&1 \
    | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip|Using cached)" \
    | grep -v "^$" \
    || true

# ── PyTorch (CPU wheel) ───────────────────────────────────────────────────────
# torch is already listed in requirements.txt but may have been satisfied by a
# version without the 'torch' import working (e.g. meta-package).  Install
# explicitly from the CPU wheel index so the binary is always present.
# On macOS the PyPI wheel already includes MPS support; the whl/cpu index works
# on both macOS and Linux and produces a smaller download on Linux.
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
        warn "PyTorch install failed — FinBERT will be unavailable (VADER still works)"
    fi
else
    ok "PyTorch already present"
fi

# Spot-check critical imports.
MISSING=()
for pkg in flask yfinance tradingview_screener pandas numpy scipy matplotlib torch transformers; do
    "$VENV_PYTHON" -c "import $pkg" 2>/dev/null || MISSING+=("$pkg")
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Could not import: ${MISSING[*]}"
    warn "Try re-running this script, or: $VENV_PIP install ${MISSING[*]}"
else
    ok "All packages verified"
fi

# ── Step 5: Launchers ─────────────────────────────────────────────────────────
step 5 "Creating launchers"

# Ensure expected output directories exist (scripts write here at runtime).
for dir in valueData growthData valuationData portfolioData plots \
           8_watchlist/watchlistData; do
    mkdir -p "$PROJECT_ROOT/$dir"
done
ok "Output directories verified"

# ── 5a. start.sh ──────────────────────────────────────────────────────────────
# Simple wrapper in the project root — works as long as the project folder
# hasn't moved. Re-running install.sh regenerates it if needed.
START_SH="$PROJECT_ROOT/start.sh"
cat > "$START_SH" <<STARTSH
#!/usr/bin/env bash
# ValuationSuite — project launcher
# Generated by install.sh on $(date "+%Y-%m-%d").
# Re-run install.sh if you move the project directory.
exec "${VENV_PYTHON}" "${PROJECT_ROOT}/app.py" "\$@"
STARTSH
chmod +x "$START_SH"
ok "start.sh written"

# ── 5b. ~/bin/valuation-suite ─────────────────────────────────────────────────
BIN_DIR="$HOME/bin"
mkdir -p "$BIN_DIR"
CLI_SCRIPT="$BIN_DIR/valuation-suite"
cat > "$CLI_SCRIPT" <<CLISCRIPT
#!/usr/bin/env bash
# valuation-suite — CLI launcher for ValuationSuite
# Generated by install.sh on $(date "+%Y-%m-%d").
# Re-run install.sh if you move the project directory.
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

# ── 5c. macOS: rebuild ValuationSuite.app ────────────────────────────────────
IS_MACOS=false
[[ "$(uname -s)" == "Darwin" ]] && IS_MACOS=true

if $IS_MACOS; then
    APP_DIR="$PROJECT_ROOT/ValuationSuite.app"
    MACOS_DIR="$APP_DIR/Contents/MacOS"
    RES_DIR="$APP_DIR/Contents/Resources"
    mkdir -p "$MACOS_DIR" "$RES_DIR"

    # Info.plist — quoted heredoc so no variable expansion inside.
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

    # Copy icon if present.
    ICNS_SRC="$PROJECT_ROOT/AppIcon.icns"
    if [[ -f "$ICNS_SRC" ]]; then
        cp "$ICNS_SRC" "$RES_DIR/AppIcon.icns"
        touch "$APP_DIR"   # nudges Finder to refresh its icon cache
        ok "App icon applied"
    else
        warn "AppIcon.icns not found — app will use default icon"
    fi

    # The launcher bakes in the absolute venv Python path.
    # Copy the .app anywhere — it will always find the correct interpreter.
    # Re-run install.sh if you move the project directory.
    cat > "$MACOS_DIR/ValuationSuite" <<LAUNCHER
#!/usr/bin/env bash
# ValuationSuite.app launcher — generated by install.sh on $(date "+%Y-%m-%d").
# Baked-in paths: venv Python and project root.
# Re-run install.sh if you move the project directory.

if [[ ! -f "${PROJECT_ROOT}/app.py" ]]; then
    osascript -e 'display alert "ValuationSuite — Project Not Found" message "The project was not found at its installed location.\n\nRe-run install.sh to rebuild this launcher." as critical' 2>/dev/null || true
    exit 1
fi

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
echo -e "  ${CYN}②${RST}  Terminal:     ${BLD}valuation-suite${RST}  (after opening a new shell)"
echo -e "  ${CYN}③${RST}  Terminal:     ${BLD}bash ${PROJECT_ROOT}/start.sh${RST}"
echo ""
echo -e "  ${BLD}Updating later:${RST}   re-run ${CYN}bash install.sh${RST} — pulls latest code, upgrades packages."
echo -e "  ${BLD}Moving the project:${RST} re-run ${CYN}bash install.sh${RST} to rebake launcher paths."
echo ""
