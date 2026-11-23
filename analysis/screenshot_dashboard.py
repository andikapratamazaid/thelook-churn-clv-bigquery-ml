"""Take real screenshots of the dashboard for the README.

Uses playwright with headless Chromium to render dashboard/index.html
and capture the models view, the decile view and the segments view.

Usage:
    python3 analysis/screenshot_dashboard.py
"""

import os

from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(BASE, "dashboard", "index.html")
OUT = os.path.join(BASE, "dashboard")


def main():
    url = "file://" + HTML
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 1000})
        page.goto(url)
        page.wait_for_timeout(600)
        page.screenshot(path=os.path.join(OUT, "dashboard_models.png"),
                        full_page=False)
        for tab, shot in (("deciles", "dashboard_deciles.png"),
                          ("segments", "dashboard_segments.png")):
            page.click(f'button[data-tab="{tab}"]')
            page.wait_for_timeout(400)
            page.screenshot(path=os.path.join(OUT, shot), full_page=False)
        browser.close()
    print("screenshots written to", OUT)


if __name__ == "__main__":
    main()
