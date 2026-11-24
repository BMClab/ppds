# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.65.2"]
# ///
"""Exercise the published WASM kernels in Chrome against a local Pages build."""

import argparse
import re
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright


@contextmanager
def editor(context, base, name):
    with context.new_page() as page:
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{base}/edit/{name}.html", wait_until="domcontentloaded")
        page.locator(".cm-editor").first.wait_for(state="attached")
        page.keyboard.press("ControlOrMeta+Shift+r")  # Run all stale cells.
        try:
            yield page
        finally:
            page.locator('[data-status="running"], [data-status="queued"]').first.wait_for(state="hidden")
            for cell in page.locator(".marimo-cell.has-error").all():
                errors.append(cell.locator('[data-cell-role="output"]').inner_text())
            if errors:
                raise AssertionError(f"{name} reported errors:\n" + "\n".join(errors))


def check(context, base):
    with editor(context, base, "participant_summary") as page:
        page.get_by_text("95 of 95 participants", exact=True).wait_for()
        page.locator("select:has(option[value='F']):has(option[value='M'])").first.select_option(label="F")
        page.get_by_text("56 of 95 participants", exact=True).wait_for()
        print("Cohort filter returned the expected 56 participants.", flush=True)

    with editor(context, base, "exploration") as page:
        page.get_by_text("4 of 4 participants", exact=True).wait_for()
        page.locator("select:has(option[value='2']):has(option[value='10'])").first.select_option(label="2")
        player = page.frame_locator('iframe[srcdoc*="Comfortable walking · Trial 2"]').first
        player.locator("#play").click()
        player.locator("#play", has_text="Pause").wait_for()
        print("Selected the second walking trial and started playback.", flush=True)

    with editor(context, base, "scans") as page:
        page.locator("marimo-dropdown select").first.select_option(label="s002")
        page.wait_for_function(
            """() => [...document.querySelectorAll('marimo-table,marimo-dataframe,table')].some(
                el => el.outerHTML.includes('38130') || el.outerHTML.includes('38,130'))""",
            timeout=150_000,
        )
        print("The selected left-foot mesh contains 38,130 triangles.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--checkout-assets", type=Path, metavar="DIR",
                        help="Serve the wheels and pressure sample from DIR when testing a pull request")
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.site.resolve()))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (sync_playwright() as playwright,
                  playwright.chromium.launch(channel="chrome") as browser,
                  browser.new_context() as context):
                context.set_default_timeout(150_000)
                served = set()
                if args.checkout_assets:
                    assets = {wheel.name: wheel for wheel in args.checkout_assets.glob("*.whl")}
                    if not assets:
                        raise ValueError(f"No wheels in {args.checkout_assets}")
                    sample = args.checkout_assets / "pressure-sample.zip"
                    assets[sample.name] = sample

                    def serve_asset(route):
                        name = route.request.url.rsplit("/", 1)[1]
                        route.fulfill(path=assets[name], headers={"Access-Control-Allow-Origin": "*"})
                        served.add(name)

                    context.route(re.compile(r"^https://raw\.githubusercontent\.com/BMClab/ppds/"
                                             r"[0-9a-f]{40}/(?:notebook-assets/)?(?:[^/]+\.whl|pressure-sample\.zip)$"),
                                  serve_asset)
                check(context, f"http://127.0.0.1:{server.server_port}")
                if args.checkout_assets and served != assets.keys():
                    raise AssertionError(f"Expected Chrome to request {sorted(assets)}, served {sorted(served)}")
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
