#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  ValuationSuite — Installer
#  Run once from any location after cloning/moving the project.
#
#  What this does:
#    1. Detects the project root (the directory containing this script)
#    2. Finds Python — prefers conda env "py311", falls back to system python3
#    3. Installs / upgrades all required packages from requirements.txt
#    4. Rebuilds ValuationSuite.app with a location-independent launcher
#       (the .app will work from any path without re-running this script)
#    5. Creates a "valuation-suite" shell command in ~/bin for terminal use
#
#  Usage:
#    chmod +x install.sh && ./install.sh
#    — or —
#    bash install.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'; BLD='\033[1m'

step()  { echo -e "\n${BLU}[${1}/${TOTAL}]${RST} ${BLD}${2}${RST}"; }
ok()    { echo -e "    ${GRN}✓${RST} ${1}"; }
warn()  { echo -e "    ${YLW}⚠${RST}  ${1}"; }
info()  { echo -e "    ${CYN}•${RST}  ${1}"; }
fail()  { echo -e "\n${RED}✗ Error:${RST} ${1}"; exit 1; }

TOTAL=4

# ── 1. Detect project root ────────────────────────────────────────────────────
step 1 "Detecting project location"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ok "Project root: $PROJECT_ROOT"

# Sanity check — make sure app.py is here
[[ -f "$PROJECT_ROOT/app.py" ]] || fail "app.py not found in $PROJECT_ROOT — run this script from the project root."

# ── 2. Find Python ────────────────────────────────────────────────────────────
step 2 "Locating Python environment"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="py311"
PYTHON_CMD=""

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" 2>/dev/null || true
    info "Conda found at $CONDA_SH"

    if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}[[:space:]]"; then
        conda activate "$CONDA_ENV" 2>/dev/null || true
        PYTHON_CMD="$(which python 2>/dev/null || echo "")"
        ok "Conda env '$CONDA_ENV' activated"
        info "Python: $PYTHON_CMD"
    else
        warn "Conda env '$CONDA_ENV' not found — will try system Python"
    fi
fi

# Fall back to system Python
if [[ -z "$PYTHON_CMD" ]]; then
    PYTHON_CMD="$(which python3 2>/dev/null || which python 2>/dev/null || echo "")"
    [[ -n "$PYTHON_CMD" ]] || fail "No Python found. Install Python 3.9+ or create a conda env named '$CONDA_ENV'."
    ok "Using system Python: $PYTHON_CMD"
fi

PYTHON_VERSION="$("$PYTHON_CMD" --version 2>&1)"
info "$PYTHON_VERSION"

# ── 3. Install packages ───────────────────────────────────────────────────────
step 3 "Installing / verifying Python packages"

REQ_FILE="$PROJECT_ROOT/requirements.txt"
[[ -f "$REQ_FILE" ]] || fail "requirements.txt not found at $REQ_FILE"

info "Running: pip install -r requirements.txt --quiet --upgrade"
PIP_OUT="$("$PYTHON_CMD" -m pip install -r "$REQ_FILE" --quiet --upgrade 2>&1 || true)"
# Surface only real errors, suppress expected noise
echo "$PIP_OUT" | grep -vE "^(Requirement already|Looking in|Collecting|Downloading|Installing|Building|WARNING: pip)" | grep -v "^$" || true

# Spot-check critical imports
MISSING=()
for pkg in flask yfinance tradingview_screener pandas numpy scipy; do
    "$PYTHON_CMD" -c "import $pkg" 2>/dev/null || MISSING+=("$pkg")
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Could not import: ${MISSING[*]}"
    warn "Try activating the correct environment and re-running this script."
else
    ok "All required packages verified"
fi

# ── 4a. Rebuild ValuationSuite.app ────────────────────────────────────────────
step 4 "Building ValuationSuite.app"

APP_DIR="$PROJECT_ROOT/ValuationSuite.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RES_DIR="$APP_DIR/Contents/Resources"

mkdir -p "$MACOS_DIR" "$RES_DIR"

# Ensure output directories exist
mkdir -p "$PROJECT_ROOT/8_watchlist/watchlistData"
ok "Output directories verified"

# ── Info.plist
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
</dict>
</plist>
PLIST

ok "Info.plist written"

# ── Launcher: bake absolute PROJECT_ROOT path in at install time ──────────────
# Using an unquoted heredoc (LAUNCHER, not 'LAUNCHER') so that ${PROJECT_ROOT}
# expands NOW, writing the real path into the script.
# Result: the .app can be copied to the Desktop, Dock, or anywhere else and
# it will always know where the project lives.
# If you ever MOVE the project folder itself, just re-run install.sh.
cat > "$MACOS_DIR/ValuationSuite" <<LAUNCHER
#!/usr/bin/env bash
# ── ValuationSuite.app launcher ──────────────────────────────────────────────
# Project path baked in by install.sh on $(date "+%Y-%m-%d").
# Copy this .app anywhere — it will always launch the correct project.
# If you move the project folder, re-run install.sh to update this path.

PROJECT_ROOT="${PROJECT_ROOT}"

# Sanity check — helpful error if the project folder was moved
if [[ ! -f "\$PROJECT_ROOT/app.py" ]]; then
    osascript -e "display alert \"ValuationSuite — Project Not Found\" message \"Expected the project at:\n\n\$PROJECT_ROOT\n\nIf you moved the project folder, re-run install.sh to rebuild this app.\" as critical" 2>/dev/null || true
    exit 1
fi

# Activate conda env if available
CONDA_SH="\$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ -f "\$CONDA_SH" ]]; then
    source "\$CONDA_SH" 2>/dev/null || true
    conda activate py311 2>/dev/null || true
fi

cd "\$PROJECT_ROOT"
exec python "\$PROJECT_ROOT/app.py"
LAUNCHER

chmod +x "$MACOS_DIR/ValuationSuite"
ok "Location-independent launcher written"

# ── 4b. Create ~/bin/valuation-suite CLI command ──────────────────────────────
BIN_DIR="$HOME/bin"
mkdir -p "$BIN_DIR"
CLI_SCRIPT="$BIN_DIR/valuation-suite"

# The CLI script bakes in the project root at install time.
# If you move the project, just re-run install.sh.
cat > "$CLI_SCRIPT" <<CLISCRIPT
#!/usr/bin/env bash
# valuation-suite — CLI launcher for ValuationSuite
# Generated by install.sh on $(date "+%Y-%m-%d")
# Project root: ${PROJECT_ROOT}
#
# Re-run install.sh if you move the project to a new location.

PROJECT_ROOT="${PROJECT_ROOT}"

CONDA_SH="\$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ -f "\$CONDA_SH" ]]; then
    source "\$CONDA_SH" 2>/dev/null || true
    conda activate py311 2>/dev/null || true
fi

cd "\$PROJECT_ROOT"
exec python "\$PROJECT_ROOT/app.py" "\$@"
CLISCRIPT

chmod +x "$CLI_SCRIPT"
ok "CLI command written: $CLI_SCRIPT"

# ── PATH reminder ─────────────────────────────────────────────────────────────
SHELL_PROFILE=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

PATH_LINE='export PATH="$HOME/bin:$PATH"'

if [[ ":$PATH:" == *":$HOME/bin:"* ]]; then
    ok "~/bin already in PATH — 'valuation-suite' command is ready"
elif [[ -n "$SHELL_PROFILE" ]]; then
    # Auto-add (idempotent — won't duplicate if already present)
    grep -qxF "$PATH_LINE" "$SHELL_PROFILE" 2>/dev/null || echo "$PATH_LINE" >> "$SHELL_PROFILE"
    ok "Added ~/bin to PATH in $SHELL_PROFILE"
    info "Run 'source $SHELL_PROFILE' or open a new terminal to use 'valuation-suite'"
else
    warn "Could not detect shell profile. Add this line manually:"
    info "$PATH_LINE"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}${BLD}✓ Installation complete!${RST}"
echo ""
echo -e "  ${BLD}How to launch:${RST}"
echo -e "  ${CYN}①${RST}  Double-click  ${BLD}ValuationSuite.app${RST}  — copy it to Desktop, Dock, or anywhere"
echo -e "  ${CYN}②${RST}  Terminal:     ${BLD}valuation-suite${RST}"
echo -e "  ${CYN}③${RST}  Terminal:     ${BLD}python ${PROJECT_ROOT}/app.py${RST}"
echo ""
echo -e "  ${BLD}Project path baked into the app:${RST} ${CYN}${PROJECT_ROOT}${RST}"
echo -e "  ${BLD}Moving the .app:${RST}     fine — copy it anywhere, it always finds the project."
echo -e "  ${BLD}Moving the project:${RST}  re-run ${CYN}bash install.sh${RST} to update the baked-in path."
echo ""
