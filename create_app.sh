#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# create_app.sh — Builds ValuationSuite.app in the project root
#
# Run this once from your activated conda environment:
#   conda activate py311
#   bash create_app.sh
#
# After running, double-click ValuationSuite.app in Finder.
# First launch: right-click → Open to bypass Gatekeeper.
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="ValuationSuite"
APP_DIR="$SCRIPT_DIR/$APP_NAME.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"

# ── Find the right Python ─────────────────────────────────────────────────────
# Priority: active conda env > python > python3
PYTHON=""
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    # Use the python from the currently activated conda environment
    PYTHON="$CONDA_PREFIX/bin/python"
elif command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "❌  No Python executable found. Activate your conda env and re-run." >&2
    exit 1
fi
echo "  Python:   $PYTHON"
echo "  Version:  $("$PYTHON" --version 2>&1)"

# ── Detect conda base (for sourcing conda.sh in the .app) ────────────────────
CONDA_BASE=""
CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/opt/anaconda3" ]; then
    CONDA_BASE="/opt/anaconda3"
fi
[ "$CONDA_ENV" = "base" ] && CONDA_ENV=""   # no need to activate base

if [ -n "$CONDA_BASE" ] && [ -n "$CONDA_ENV" ]; then
    echo "  Conda:    $CONDA_BASE  (env: $CONDA_ENV)"
fi

# ── Build .app directory structure ────────────────────────────────────────────
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR"
mkdir -p "$APP_DIR/Contents/Resources"

# ── Embed launcher script ─────────────────────────────────────────────────────
# Strategy A — conda env active: source conda.sh + activate → python launcher.py
# Strategy B — no conda or base: use the exact Python binary path directly
{
    printf '#!/usr/bin/env bash\n'
    if [ -n "$CONDA_BASE" ] && [ -n "$CONDA_ENV" ]; then
        printf '# Activate conda env "%s" then launch\n' "$CONDA_ENV"
        printf 'source "%s/etc/profile.d/conda.sh" 2>/dev/null || true\n' "$CONDA_BASE"
        printf 'conda activate "%s" 2>/dev/null || true\n' "$CONDA_ENV"
        printf 'cd "%s"\n' "$SCRIPT_DIR"
        printf 'python "%s/launcher.py"\n' "$SCRIPT_DIR"
    else
        printf '# Launch with explicit Python path\n'
        printf 'cd "%s"\n' "$SCRIPT_DIR"
        printf '"%s" "%s/launcher.py"\n' "$PYTHON" "$SCRIPT_DIR"
    fi
} > "$MACOS_DIR/$APP_NAME"
chmod +x "$MACOS_DIR/$APP_NAME"

echo ""
echo "  Embedded launcher:"
echo "  ──────────────────"
cat "$MACOS_DIR/$APP_NAME"
echo "  ──────────────────"

# ── Info.plist ────────────────────────────────────────────────────────────────
cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>     <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>     <string>com.valuationsuite.launcher</string>
  <key>CFBundleName</key>           <string>Valuation Suite</string>
  <key>CFBundleDisplayName</key>    <string>Valuation Suite</string>
  <key>CFBundleVersion</key>        <string>1.0</string>
  <key>CFBundlePackageType</key>    <string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key> <string>10.13</string>
</dict>
</plist>
PLIST

echo ""
echo "  ✓  $APP_NAME.app created successfully."
echo ""
echo "  → Drag ValuationSuite.app to your Applications folder (optional)."
echo "  → Double-click to launch."
echo "  → First launch: right-click → Open to bypass Gatekeeper."
