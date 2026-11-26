"""Check that the committed manuscript page records a clean run of the current notebook."""

import hashlib

from ppds.paths import ROOT

NOTEBOOK = ROOT / "paper" / "data_descriptor.py"
PAGE = NOTEBOOK.with_suffix(".html")
REGENERATE = (
    "Regenerate it with "
    "`uv run --locked marimo export html paper/data_descriptor.py -o paper/data_descriptor.html`."
)


def test_saved_page_was_generated_from_this_notebook():
    # The notebook prints the SHA-256 of its own source in the run record.
    digest = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    assert digest in PAGE.read_text(encoding="utf-8"), f"{PAGE.name} is out of date. {REGENERATE}"


def test_saved_page_has_no_cell_errors():
    # marimo still writes the page when a cell fails, storing the failure as an error output.
    assert '"type": "error"' not in PAGE.read_text(encoding="utf-8"), f"{PAGE.name} has cell errors."
