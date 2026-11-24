# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.65.2"]
# ///
"""Exercise the published WASM kernels in Chrome against a local Pages build."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright


def open_editor(page, base, name):
    page.goto(f"{base}/edit/{name}.html", wait_until="domcontentloaded")
    page.locator(".cm-editor").first.wait_for(state="attached")
    page.keyboard.press("Control+Shift+r")  # Run all stale cells in the editor.


def check(browser, base):
    page = browser.new_page()
    page.set_default_timeout(150_000)
    try:
        open_editor(page, base, "participant_summary")
        page.get_by_text("95 of 95 participants", exact=True).wait_for()
        page.locator("select:has(option[value='F']):has(option[value='M'])").first.select_option(label="F")
        page.get_by_text("56 of 95 participants", exact=True).wait_for()
        print("Cohort filter returned the expected 56 participants.", flush=True)

        open_editor(page, base, "exploration")
        page.get_by_text("4 of 4 participants", exact=True).wait_for()
        page.locator("select:has(option[value='2']):has(option[value='10'])").first.select_option(label="2")
        player = page.frame_locator('iframe[srcdoc*="Comfortable walking · Trial 2"]').first
        player.locator("#play").click()
        player.locator("#play", has_text="Pause").wait_for()
        print("Selected the second walking trial and started playback.", flush=True)

        open_editor(page, base, "scans")
        page.locator("marimo-dropdown select").first.select_option(label="s002")
        page.wait_for_function(
            """() => [...document.querySelectorAll('marimo-table,marimo-dataframe,table')].some(
                el => el.outerHTML.includes('38130') || el.outerHTML.includes('38,130'))""",
            timeout=150_000,
        )
        print("The selected left-foot mesh contains 38,130 triangles.", flush=True)
    finally:
        page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.site.resolve()))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (sync_playwright() as playwright,
                  playwright.chromium.launch(channel="chrome") as browser):
                check(browser, f"http://127.0.0.1:{server.server_port}")
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
