# Notebook runtime assets

The examples install the same two pure-Python wheels in GitHub Pages, marimo.app and molab. Their script headers point to immutable Git commits so a saved notebook can reopen its original runtime.

## Update the wheels

1. After changing either package or its runtime dependencies, build both wheels from the repository root.

   ```sh
   uv build --all-packages --wheel --out-dir dist/wheels
   ```

2. Copy the two wheels printed by the build to `notebook-assets/`. Remove any superseded wheel filenames from that directory. Commit the package changes and wheels, then get the full commit ID with `git rev-parse HEAD`.
3. In the script headers of the notebooks in `examples/` and `paper/`, set both wheel URLs to `https://raw.githubusercontent.com/BMClab/ppds/<full-commit-id>/notebook-assets/<wheel-filename>`. The saved page `paper/data_descriptor.html` records the SHA-256 of its notebook, so after changing that header regenerate the page with `uv run --locked marimo export html paper/data_descriptor.py -o paper/data_descriptor.html` (a full run with the complete release). The browser dashboard embeds its header too, so also rebuild `examples/dashboard/dashboard.html` with `uv run --locked python examples/dashboard/export_html.py`. Commit the pins and the regenerated pages together, separately from the wheels, so the pins can reference the preceding commit. Preserve the asset commit when merging.
4. Run `uv run --locked pytest -q tests/test_notebook_assets.py` in a full clone. It compares a fresh build with the stored package files and runtime metadata, then verifies that the notebook URLs resolve to the same Git blobs. Push the two commits together.

To test a local Pages build with unpublished assets, run `uv run --no-project --script scripts/check_browser_examples.py _site --checkout-assets notebook-assets`. Pull requests use this mode. The check on `main` fetches the published URLs.

The 9.0 MB `pressure-sample.zip` is a separate asset. Rebuild it only when the four-participant sample changes. For a replacement, commit the ZIP first, update its immutable URL and SHA-256 in `src/ppds/_browser_sample.py`, then rebuild and re-pin the wheels.
