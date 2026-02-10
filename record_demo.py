"""
Screen-record a polished VayuChat demo with:
- Animated cursor with click ripples
- Mixed interactions: welcome buttons, sidebar clicks, typed queries
- Smooth scrolling to keep responses in view
- Tighter pacing between queries
- Trimmed loading screen (only shows loaded app)
- High resolution (2560x1440)
- Optional background music overlay (bg_music.mp3)

Usage:
    1. Start the app:  streamlit run app.py --server.port 8502
    2. Run this:       python record_demo.py
    3. Output:         vayuchat_demo.mp4

Text overlays and music are added in post-processing via ffmpeg.
Place bg_music.mp3 in this directory for background music.
"""
import time
import os
from playwright.sync_api import sync_playwright

URL = "http://localhost:8502"
VIDEO_DIR = "/tmp/vayuchat_recording"
OUTPUT = os.path.join(os.path.dirname(__file__), "vayuchat_demo.mp4")
MUSIC_PATH = os.path.join(os.path.dirname(__file__), "bg_music.mp3")
RES_W, RES_H = 2560, 1440

os.makedirs(VIDEO_DIR, exist_ok=True)


# ── Cursor injection ───────────────────────────────────────────────────────

INJECT_JS = """() => {
    if (!document.getElementById('fc')) {
        const c = document.createElement('div');
        c.id = 'fc';
        c.innerHTML = '<svg width="24" height="30" viewBox="0 0 24 28"><path d="M5 2v22l5.5-5.5 4.5 8.5 3-1.5-4.5-8.5H21z" fill="#111" stroke="#fff" stroke-width="1.2"/></svg>';
        c.style.cssText = 'position:fixed;top:-40px;left:-40px;z-index:999999;pointer-events:none;filter:drop-shadow(1px 2px 3px rgba(0,0,0,0.35));will-change:transform;';
        document.body.appendChild(c);
    }
    if (!document.getElementById('fc-style')) {
        const s = document.createElement('style');
        s.id = 'fc-style';
        s.textContent = `
            @keyframes clickPulse {
                0%   { transform: scale(0.3); opacity: 0.9; }
                100% { transform: scale(2.0); opacity: 0; }
            }
        `;
        document.head.appendChild(s);
    }
}"""


def inject(page):
    page.evaluate(INJECT_JS)


# ── Cursor helpers ─────────────────────────────────────────────────────────

def move_cursor(page, x, y, ms=400):
    page.evaluate(f"""() => {{
        const c = document.getElementById('fc');
        if (!c) return;
        c.style.transition = 'transform {ms}ms cubic-bezier(.25,.1,.25,1)';
        c.style.transform = 'translate({x}px,{y}px)';
    }}""")
    time.sleep(ms / 1000 + 0.05)


def click_xy(page, x, y):
    move_cursor(page, x, y)
    page.evaluate(f"""() => {{
        const r = document.createElement('div');
        r.style.cssText = 'position:fixed;left:{x - 14}px;top:{y - 14}px;width:28px;height:28px;border-radius:50%;border:2.5px solid #3b82f6;z-index:999998;pointer-events:none;animation:clickPulse 0.45s ease-out forwards;';
        document.body.appendChild(r);
        setTimeout(() => r.remove(), 500);
    }}""")
    time.sleep(0.08)
    page.mouse.click(x, y)
    time.sleep(0.25)


def click_el(page, loc):
    box = loc.bounding_box()
    if not box:
        loc.click()
        return
    click_xy(page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))


# ── Interaction methods ────────────────────────────────────────────────────

def type_query(page, text, char_delay=0.035):
    """Type into chat input char-by-char and press Enter."""
    textarea = page.locator('[data-testid="stChatInput"] textarea')
    box = textarea.bounding_box()
    if box:
        click_xy(page, int(box["x"] + 180), int(box["y"] + box["height"] / 2))
    time.sleep(0.15)
    for ch in text:
        page.keyboard.type(ch, delay=0)
        time.sleep(char_delay)
    time.sleep(0.3)
    page.keyboard.press("Enter")
    time.sleep(0.15)


def click_sidebar_btn(page, section, query):
    """Click a sidebar button, expanding its section if collapsed."""
    sidebar = page.locator('[data-testid="stSidebar"]')
    # Expand section if needed
    expander = sidebar.locator("details").filter(has_text=section)
    if expander.count() > 0:
        expander.first.scroll_into_view_if_needed()
        time.sleep(0.2)
        if expander.first.get_attribute("open") is None:
            click_el(page, expander.first.locator("summary"))
            time.sleep(0.5)
    # Click button
    btn = sidebar.locator("button").filter(has_text=query)
    if btn.count() > 0:
        btn.first.scroll_into_view_if_needed()
        time.sleep(0.2)
        click_el(page, btn.first)
    else:
        print(f"  WARNING: sidebar button not found: {query[:50]}...")


def click_welcome_btn(page, label):
    """Click a welcome screen capability button."""
    btn = page.locator("button").filter(has_text=label)
    if btn.count() > 0:
        click_el(page, btn.first)
    else:
        print(f"  WARNING: welcome button not found: {label}")


# ── Response handling ──────────────────────────────────────────────────────

def wait_response(page, timeout=60):
    """Wait for Streamlit processing to complete."""
    try:
        page.wait_for_selector("text=Processing with", timeout=8000)
    except Exception:
        pass
    try:
        page.wait_for_function(
            "() => !document.body.innerText.includes('Processing with')",
            timeout=timeout * 1000,
        )
    except Exception:
        pass
    time.sleep(1.0)
    inject(page)


def scroll_to_response(page):
    """Scroll so the latest response is fully visible."""
    # First scroll to bottom to ensure the response area is in view
    page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
    time.sleep(0.8)
    # Then try to center the last response element
    page.evaluate("""() => {
        const els = document.querySelectorAll(
            '.assistant-message, .js-plotly-plot, [data-testid="stDataFrame"], [data-testid="stImage"]'
        );
        if (els.length) {
            const el = els[els.length - 1];
            const rect = el.getBoundingClientRect();
            // If element is taller than viewport, scroll to its top
            if (rect.height > window.innerHeight * 0.7) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }""")
    time.sleep(1.0)


def expand_code(page, hold=2.0):
    """Expand and then collapse the latest code expander."""
    exps = page.locator("details summary").filter(has_text="View Generated Code")
    n = exps.count()
    if n > 0:
        target = exps.nth(n - 1)
        target.scroll_into_view_if_needed()
        time.sleep(0.3)
        click_el(page, target)
        time.sleep(hold)
        click_el(page, target)
        time.sleep(0.3)


# ── Demo sequence ──────────────────────────────────────────────────────────
# method: "welcome" | "sidebar" | "type"

DEMOS = [
    # 1. Welcome button → Interactive Map (star feature opener)
    dict(method="welcome", button="Interactive Maps",
         code=False, hold=3.5),
    # 2. Type → Text answer (fast FC path)
    dict(method="type", query="Which city has the highest PM2.5 in 2023?",
         code=True, hold=2.0),
    # 3. Sidebar click → Trend chart
    dict(method="sidebar", section="Getting Started",
         query="Plot yearly PM2.5 trends from 2020 to 2024",
         code=True, hold=2.5),
    # 4. Type → Data table
    dict(method="type", query="Rank top 10 most polluted cities",
         code=False, hold=2.5),
    # 5. Sidebar click → Change map
    dict(method="sidebar", section="Maps & Geography",
         query="Show pollution change from 2020 to 2023 on a map",
         code=False, hold=3.0),
    # 6. Type → Weather impact chart
    dict(method="type", query="How does wind speed affect PM2.5?",
         code=True, hold=2.5),
]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": RES_W, "height": RES_H},
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": RES_W, "height": RES_H},
        )
        page = ctx.new_page()

        # ── Load app ──────────────────────────────────────────
        print("Loading app...")
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector(
            '[data-testid="stChatInput"] textarea', timeout=20000
        )
        time.sleep(2.5)

        # Mark when demo content starts (for trimming)
        t_start = time.time()
        inject(page)
        time.sleep(0.3)

        # Brief welcome screen
        print("Welcome screen...")
        time.sleep(1.5)

        # ── Run demo queries ──────────────────────────────────
        for i, d in enumerate(DEMOS):
            label = d.get("query", d.get("button", ""))
            print(f"[{i + 1}/{len(DEMOS)}] {label}")

            # Send the query
            if d["method"] == "welcome":
                click_welcome_btn(page, d["button"])
            elif d["method"] == "sidebar":
                click_sidebar_btn(page, d["section"], d["query"])
            elif d["method"] == "type":
                type_query(page, d["query"])

            # Wait for response
            wait_response(page)

            # Scroll to show response
            scroll_to_response(page)

            # Hold to view the result
            time.sleep(d["hold"])

            # Show code expander if requested
            if d["code"]:
                expand_code(page)

            time.sleep(0.2)

        # ── Final scroll through conversation ─────────────────
        print("Final scroll...")
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(1.0)
        page.evaluate(
            "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"
        )
        time.sleep(1.5)

        # ── Save recording ────────────────────────────────────
        t_demo = time.time() - t_start
        print(f"Demo actions: {t_demo:.0f}s")
        page.close()
        ctx.close()
        browser.close()

    # ── Post-process: convert, trim, add music ────────────────────────────

    # Find webm
    webm = None
    for f in os.listdir(VIDEO_DIR):
        if f.endswith(".webm"):
            webm = os.path.join(VIDEO_DIR, f)
            break
    if not webm:
        print("ERROR: no recording found")
        return

    # Calculate trim offset (remove loading screen)
    raw_dur = float(
        os.popen(
            f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{webm}"'
        ).read().strip()
    )
    trim_ss = max(0, raw_dur - t_demo - 0.5)
    print(f"Raw {raw_dur:.0f}s, trimming first {trim_ss:.1f}s")

    # Build ffmpeg command
    has_music = os.path.exists(MUSIC_PATH)
    cmd = f'ffmpeg -y -ss {trim_ss:.2f} -i "{webm}" '
    if has_music:
        cmd += f'-i "{MUSIC_PATH}" '
    cmd += (
        "-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p "
        "-movflags +faststart "
    )
    if has_music:
        fade_out = max(1, t_demo - 2)
        cmd += (
            f'-filter_complex "[1:a]volume=0.12,afade=t=out:st={fade_out:.0f}:d=2[a]" '
            f"-map 0:v -map \"[a]\" -shortest "
        )
    cmd += f'"{OUTPUT}" 2>/dev/null'

    print("Encoding...")
    os.system(cmd)
    os.remove(webm)

    if os.path.exists(OUTPUT):
        mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        dur = float(
            os.popen(
                f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{OUTPUT}"'
            ).read().strip()
        )
        print(f"\nDone! {OUTPUT}")
        print(f"  {dur:.0f}s, {mb:.1f} MB, {RES_W}x{RES_H}")
        if not has_music:
            print("  (no bg_music.mp3 found — video has no audio)")
            print("  To add music later: ffmpeg -i vayuchat_demo.mp4 -i bg_music.mp3 "
                  '-filter_complex "[1:a]volume=0.12[a]" -map 0:v -map "[a]" '
                  "-shortest -c:v copy output.mp4")
    else:
        print("ERROR: video not created")
        for f in os.listdir(VIDEO_DIR):
            print(f"  found: {f}")


if __name__ == "__main__":
    main()
