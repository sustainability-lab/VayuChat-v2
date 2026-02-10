"""
Generate a VayuChat demo video showing different output types.
Creates frames as images, then combines with ffmpeg.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.io as pio

# ── Config ──────────────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS = 30
FADE_FRAMES = int(0.4 * FPS)  # 0.4s crossfade
FRAME_DIR = "/tmp/vayuchat_frames"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "vayuchat_demo.mp4")

# Colors
WHITE = (255, 255, 255)
BG = (248, 250, 252)         # slate-50
BRAND = (37, 99, 235)        # blue-600
BRAND_DARK = (15, 23, 42)    # slate-900
BRAND_LIGHT = (219, 234, 254)  # blue-100
GRAY = (100, 116, 139)       # slate-500
LIGHT_GRAY = (226, 232, 240) # slate-200

# Fonts
def _font(size, bold=False):
    """Load Helvetica (macOS) or fallback."""
    try:
        idx = 1 if bold else 0  # Helvetica.ttc: 0=regular, 1=bold
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=idx)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
        except Exception:
            return ImageFont.load_default()

FONT_TITLE = _font(64, bold=True)
FONT_SUBTITLE = _font(28)
FONT_QUERY = _font(26, bold=True)
FONT_BODY = _font(22)
FONT_SMALL = _font(18)
FONT_BADGE = _font(16, bold=True)

os.makedirs(FRAME_DIR, exist_ok=True)


# ── Drawing helpers ─────────────────────────────────────────────────────────

def new_frame(bg=BG):
    return Image.new("RGB", (W, H), bg)


def draw_header(draw, query_text=None):
    """Draw top bar with VayuChat branding."""
    # Blue bar
    draw.rectangle([(0, 0), (W, 70)], fill=BRAND)
    draw.text((40, 14), "VayuChat", fill=WHITE, font=_font(32, bold=True))
    draw.text((260, 22), "AI Air Quality Analysis", fill=BRAND_LIGHT, font=_font(20))

    if query_text:
        # User query bubble
        y = 100
        draw.rounded_rectangle(
            [(W - 60 - len(query_text) * 12, y), (W - 40, y + 50)],
            radius=12, fill=BRAND,
        )
        draw.text(
            (W - 50 - len(query_text) * 12, y + 12),
            query_text, fill=WHITE, font=FONT_QUERY,
        )
        return y + 70  # content start y
    return 90


def draw_badge(draw, x, y, text, color=BRAND):
    """Draw a small badge/pill."""
    tw = len(text) * 10 + 16
    draw.rounded_rectangle([(x, y), (x + tw, y + 26)], radius=13, fill=color)
    draw.text((x + 8, y + 4), text, fill=WHITE, font=FONT_BADGE)
    return tw


def paste_centered(base, overlay, y_offset, max_w=None, max_h=None):
    """Paste an image centered on base at y_offset, scaling if needed."""
    ow, oh = overlay.size
    mw = max_w or (W - 120)
    mh = max_h or (H - y_offset - 60)
    scale = min(mw / ow, mh / oh, 1.0)
    if scale < 1:
        overlay = overlay.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)
    ow, oh = overlay.size
    x = (W - ow) // 2
    base.paste(overlay, (x, y_offset))


def wrap_text(text, font, max_width, draw):
    """Word-wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


# ── Generate real outputs ───────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv("AQ_met_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
ncap_df = pd.read_csv("ncap_funding_data.csv")

from analysis_tools import (
    get_statistics, df_ranking,
    plot_trend, plot_comparison, plot_correlation, plot_met_impact,
    map_pollution, map_change,
)

print("Generating outputs...")

# 1. Text answer
text_result = get_statistics(df, pollutant="PM2.5", group_by="City", top_n=5, year=2023)

# 2. Trend chart
trend_path = plot_trend(df, pollutant="PM2.5", freq="yearly", year_start=2019, year_end=2024)

# 3. Comparison chart
comp_path = plot_comparison(df, pollutant="PM2.5", comparison_type="seasonal")

# 4. Table
table_result = df_ranking(df, pollutant="PM2.5", group_by="City", top_n=10)

# 5. Map
map_fig = map_pollution(df, pollutant="PM2.5", year=2023, top_n=50)
map_path = "/tmp/vayuchat_map.png"
pio.write_image(map_fig, map_path, width=1400, height=750, scale=2)

# 6. Change map
change_fig = map_change(df, pollutant="PM2.5", year_start=2019, year_end=2024)
change_path = "/tmp/vayuchat_change.png"
pio.write_image(change_fig, change_path, width=1400, height=750, scale=2)

# 7. Met impact
met_path = plot_met_impact(df, met_factor="wind speed", pollutant="PM2.5")

print("Building slides...")


# ── Slide builders ──────────────────────────────────────────────────────────

def slide_title():
    """Opening title slide."""
    img = new_frame(BRAND_DARK)
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((W // 2, 300), "VayuChat", fill=WHITE, font=_font(96, bold=True), anchor="mm")
    draw.text((W // 2, 390), "AI-Powered Air Quality Analysis for India",
              fill=BRAND_LIGHT, font=_font(32), anchor="mm")

    # Divider
    draw.line([(W // 2 - 120, 440), (W // 2 + 120, 440)], fill=BRAND, width=3)

    # Stats
    stats = ["288 Cities", "2017-2024", "9 Pollutants", "5 Met Variables"]
    sx = W // 2 - 350
    for i, s in enumerate(stats):
        x = sx + i * 190
        draw.text((x, 490), s, fill=LIGHT_GRAY, font=_font(22, bold=True), anchor="lm")

    # Subtitle
    draw.text((W // 2, 600), "Sustainability Lab  |  IIT Gandhinagar",
              fill=GRAY, font=_font(22), anchor="mm")

    return img


def slide_text_answer():
    """Text output demo."""
    img = new_frame()
    draw = ImageDraw.Draw(img)
    cy = draw_header(draw, "Which are the top 5 most polluted cities in 2023?")

    # Badge
    draw_badge(draw, 60, cy + 10, "TEXT ANSWER", BRAND)

    # Response box
    ry = cy + 55
    draw.rounded_rectangle([(50, ry), (W - 50, H - 50)], radius=16, fill=WHITE, outline=LIGHT_GRAY)

    # Render text response
    lines = text_result.split("\n")
    ty = ry + 25
    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            # Header
            clean = line.strip("*").strip()
            draw.text((80, ty), clean, fill=BRAND_DARK, font=_font(22, bold=True))
            ty += 36
        elif line.startswith("- "):
            draw.text((80, ty), line, fill=BRAND_DARK, font=FONT_BODY)
            ty += 32
        elif line.strip():
            wrapped = wrap_text(line, FONT_BODY, W - 200, draw)
            for wl in wrapped:
                draw.text((80, ty), wl, fill=BRAND_DARK, font=FONT_BODY)
                ty += 30
        else:
            ty += 12

    return img


def slide_chart(chart_path, query, badge_text="CHART"):
    """Chart output demo."""
    img = new_frame()
    draw = ImageDraw.Draw(img)
    cy = draw_header(draw, query)

    draw_badge(draw, 60, cy + 10, badge_text, (220, 38, 38))  # red badge

    # Load and paste chart
    chart = Image.open(chart_path)
    paste_centered(img, chart, cy + 50)
    return img


def slide_table():
    """Table output demo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = new_frame()
    draw = ImageDraw.Draw(img)
    cy = draw_header(draw, "Rank top 10 most polluted cities")

    draw_badge(draw, 60, cy + 10, "TABLE", (5, 150, 105))  # green badge

    # Render table with matplotlib
    tdf = table_result.copy()
    fig_t, ax_t = plt.subplots(figsize=(16, 5))
    ax_t.axis("off")
    cols = list(tdf.columns)
    cell_text = tdf.values.tolist()
    table = ax_t.table(
        cellText=cell_text, colLabels=cols, loc="center",
        cellLoc="center", colColours=[(0.145, 0.388, 0.922, 0.15)] * len(cols),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor((0.886, 0.910, 0.941))
        if key[0] == 0:
            cell.set_text_props(weight="bold", color="#1e293b")
            cell.set_facecolor("#dbeafe")
    fig_t.tight_layout()
    table_path = "/tmp/vayuchat_table.png"
    fig_t.savefig(table_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig_t)

    table_img = Image.open(table_path)
    paste_centered(img, table_img, cy + 55)
    return img


def slide_map(map_img_path, query):
    """Map output demo."""
    img = new_frame()
    draw = ImageDraw.Draw(img)
    cy = draw_header(draw, query)

    draw_badge(draw, 60, cy + 10, "INTERACTIVE MAP", (124, 58, 237))  # purple badge

    map_img = Image.open(map_img_path)
    paste_centered(img, map_img, cy + 50)
    return img


def slide_closing():
    """Closing slide."""
    img = new_frame(BRAND_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((W // 2, 340), "VayuChat", fill=WHITE, font=_font(72, bold=True), anchor="mm")
    draw.text((W // 2, 420), "Try it yourself", fill=BRAND_LIGHT, font=_font(32), anchor="mm")

    # Feature list
    features = [
        "Text Answers  |  Interactive Tables  |  Charts & Plots  |  Interactive Maps"
    ]
    for i, f in enumerate(features):
        draw.text((W // 2, 510 + i * 40), f, fill=LIGHT_GRAY, font=_font(24), anchor="mm")

    draw.text((W // 2, 650), "Sustainability Lab  |  IIT Gandhinagar",
              fill=GRAY, font=_font(22), anchor="mm")

    return img


# ── Build all slides ────────────────────────────────────────────────────────

slides = [
    (slide_title(), 3.5),                                                             # 3.5s
    (slide_text_answer(), 5.0),                                                       # 5s
    (slide_chart(trend_path, "Plot yearly PM2.5 trends from 2019 to 2024"), 4.5),     # 4.5s
    (slide_chart(comp_path, "Compare PM2.5 across seasons", "COMPARISON"), 4.0),      # 4s
    (slide_table(), 4.5),                                                             # 4.5s
    (slide_map(map_path, "Show top 50 polluted cities on a map for 2023"), 5.0),      # 5s
    (slide_map(change_path, "Show pollution change from 2019 to 2024 on a map"), 5.0),# 5s
    (slide_chart(met_path, "How does wind speed affect PM2.5?", "MET IMPACT"), 4.0),  # 4s
    (slide_closing(), 3.5),                                                           # 3.5s
]

# Total: ~39 seconds

# ── Generate frames with crossfades ─────────────────────────────────────────

print("Generating frames...")
frame_num = 0

for slide_idx, (slide_img, duration) in enumerate(slides):
    hold_frames = int(duration * FPS) - FADE_FRAMES  # hold before fade starts

    # Hold frames
    for _ in range(hold_frames):
        slide_img.save(os.path.join(FRAME_DIR, f"frame_{frame_num:05d}.png"))
        frame_num += 1

    # Crossfade to next slide
    if slide_idx < len(slides) - 1:
        next_img = slides[slide_idx + 1][0]
        for f in range(FADE_FRAMES):
            alpha = f / FADE_FRAMES
            blended = Image.blend(slide_img, next_img, alpha)
            blended.save(os.path.join(FRAME_DIR, f"frame_{frame_num:05d}.png"))
            frame_num += 1
    else:
        # Last slide: just hold
        for _ in range(FADE_FRAMES):
            slide_img.save(os.path.join(FRAME_DIR, f"frame_{frame_num:05d}.png"))
            frame_num += 1

print(f"Total frames: {frame_num} ({frame_num / FPS:.1f}s)")

# ── Encode video ────────────────────────────────────────────────────────────

print("Encoding video...")
cmd = (
    f"ffmpeg -y -framerate {FPS} "
    f"-i {FRAME_DIR}/frame_%05d.png "
    f"-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
    f"-movflags +faststart "
    f'"{OUTPUT_PATH}"'
)
os.system(cmd)

# Cleanup frames
import shutil
shutil.rmtree(FRAME_DIR, ignore_errors=True)

# Cleanup generated plots
for p in [trend_path, comp_path, met_path, map_path, change_path, "/tmp/vayuchat_table.png"]:
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass

if os.path.exists(OUTPUT_PATH):
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nDone! Video saved to: {OUTPUT_PATH} ({size_mb:.1f} MB)")
else:
    print("ERROR: Video was not created")
