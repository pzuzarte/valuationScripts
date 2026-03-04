#!/usr/bin/env python3
"""
create_icon.py — Generate ValuationSuite.app icon (1024×1024 PNG).

Design:
  • Deep crimson → dark-red radial gradient background
  • White ascending bar chart (4 bars, tallest on right)
  • Thin white upward-trend line over the bars
  • Subtle white "VS" monogram, bottom-right corner

Output: AppIcon.iconset/icon_1024x1024.png
Run:    python create_icon.py
"""

import os, math
from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
OUT_DIR = os.path.join(os.path.dirname(__file__), "AppIcon.iconset")
OUT_FILE = os.path.join(OUT_DIR, "icon_1024x1024.png")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Background — radial gradient (centre bright crimson → edge deep red) ──
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
pixels = img.load()

CX, CY = SIZE / 2, SIZE / 2
R_MAX   = math.hypot(CX, CY)

# Crimson at centre (#C8102E), deep wine at edge (#5C0011)
C_INNER = (200, 16,  46)
C_OUTER = ( 92,  0,  17)

for y in range(SIZE):
    for x in range(SIZE):
        d = math.hypot(x - CX, y - CY) / R_MAX          # 0..~1
        t = min(d, 1.0)
        r = int(C_INNER[0] + (C_OUTER[0] - C_INNER[0]) * t)
        g = int(C_INNER[1] + (C_OUTER[1] - C_INNER[1]) * t)
        b = int(C_INNER[2] + (C_OUTER[2] - C_INNER[2]) * t)
        pixels[x, y] = (r, g, b, 255)

draw = ImageDraw.Draw(img)

# ── 2. Rounded-rectangle clip mask for iOS-style shape ─────────────────────
mask = Image.new("L", (SIZE, SIZE), 0)
md   = ImageDraw.Draw(mask)
RADIUS = 220
md.rounded_rectangle([(0, 0), (SIZE - 1, SIZE - 1)], radius=RADIUS, fill=255)
img.putalpha(mask)

draw = ImageDraw.Draw(img)

# ── 3. Ascending bar chart ───────────────────────────────────────────────────
# 4 bars, left to right, heights 30 % → 46 % → 63 % → 82 % of canvas
BAR_HEIGHTS_PCT = [0.30, 0.46, 0.63, 0.82]
N_BARS  = len(BAR_HEIGHTS_PCT)
BAR_W   = 115          # bar width  (px)
GAP     = 42           # gap between bars
CHART_Y = 620          # baseline y  (bars sit above this)
LEFT_X  = 160          # x of leftmost bar

bar_tops = []
for i, h_pct in enumerate(BAR_HEIGHTS_PCT):
    x0 = LEFT_X + i * (BAR_W + GAP)
    x1 = x0 + BAR_W
    bar_h = int(h_pct * SIZE * 0.68)     # scale so tallest bar ≈ 68 % of icon
    y0 = CHART_Y - bar_h
    y1 = CHART_Y

    # Bar with rounded top
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=16,
                            fill=(255, 255, 255, 235))

    bar_tops.append((x0 + BAR_W // 2, y0))   # top-centre of each bar

# ── 4. Trend line connecting bar tops ────────────────────────────────────────
# Extend slightly beyond first and last bar for a clean look
lw = 14   # line width
for i in range(len(bar_tops) - 1):
    draw.line([bar_tops[i], bar_tops[i + 1]], fill=(255, 255, 255, 255), width=lw)

# Arrow head at the last bar top
ax, ay = bar_tops[-1]
draw.polygon(
    [(ax, ay - 28), (ax - 22, ay + 10), (ax + 22, ay + 10)],
    fill=(255, 255, 255, 255)
)

# ── 5. "VS" monogram ─────────────────────────────────────────────────────────
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 88)
except Exception:
    font = ImageFont.load_default()

MONO = "VS"
bbox = draw.textbbox((0, 0), MONO, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
tx = SIZE - tw - 62
ty = SIZE - th - 62
draw.text((tx, ty), MONO, font=font, fill=(255, 255, 255, 155))

# ── 6. Save ──────────────────────────────────────────────────────────────────
img.save(OUT_FILE, "PNG")
print(f"  Icon written → {OUT_FILE}")
