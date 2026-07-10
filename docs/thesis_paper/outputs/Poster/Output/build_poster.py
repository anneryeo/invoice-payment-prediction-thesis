"""Render poster.html to an A0-portrait PDF using headless Chromium.

Usage:
    python build_poster.py

Requires:
    pip install playwright
    playwright install chromium
"""

import pathlib

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent.resolve()
HTML_PATH = HERE / "poster.html"
PDF_PATH = HERE / "thesis_poster.pdf"

# A0 portrait, matching both sample reference posters.
PAGE_WIDTH_MM = 841
PAGE_HEIGHT_MM = 1189


def build():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(HTML_PATH.as_uri())
        page.wait_for_timeout(300)  # let inline SVG/webfonts settle
        page.pdf(
            path=str(PDF_PATH),
            width=f"{PAGE_WIDTH_MM}mm",
            height=f"{PAGE_HEIGHT_MM}mm",
            print_background=True,
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
        )
        browser.close()
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    build()
