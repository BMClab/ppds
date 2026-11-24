"""Exercise notebook discovery and rebuilds with the real marimo exporter."""

from pathlib import Path
from subprocess import CalledProcessError

import pytest


def test_pages_rebuilds_executes_and_indexes_examples(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts import build_pages

    monkeypatch.setattr(build_pages, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr(build_pages, "GUIDES", (Path("README.md"), build_pages.CATALOG))
    examples = tmp_path / "examples"
    examples.mkdir()
    (tmp_path / "README.md").write_text("# PPDS\n\n[Examples](examples/README.md)")
    (examples / "README.md").write_text("# Examples\n\n[Demo](demo.py)")
    (examples / "helper.py").write_text('"""Docs may mention marimo.App()."""')
    template = """import marimo

app = marimo.App()


@app.cell
def _():
    {code}
    return
"""
    demo = examples / "demo.py"
    demo.write_text(template.format(code='print("saved", 6 * 7)'))
    build_pages.main()
    site = tmp_path / "_site"
    assert 'href="examples/index.html"' in (site / "index.html").read_text()
    page = site / "notebooks/demo.html"
    assert "saved 42" in page.read_text()
    editor = site / "edit/demo.html"
    assert "marimo" in editor.read_text()
    assert 'href="../edit/demo.html"' in page.read_text()
    ref = f"BMClab/ppds/blob/{'a' * 40}/examples/demo.py"
    assert f'href="https://molab.marimo.io/github/{ref}"' in page.read_text()
    assert f'href="https://marimo.app/github.com/{ref}"' in page.read_text()
    for subpage in (site / "examples/index.html", page):
        html = subpage.read_text()
        assert 'href="../index.html"' in html
        assert 'href="https://github.com/BMClab/ppds"' in html
    assert not (site / "notebooks/helper.html").exists()

    demo.write_text(template.format(code='print("saved", 7 * 7)'))
    new = examples / "another.py"
    new.write_text(template.format(code='print("new example")'))
    build_pages.main()
    updated = page.read_text()
    assert "saved 49" in updated and "saved 42" not in updated
    assert (site / "notebooks/demo.py").read_bytes() == demo.read_bytes()
    assert (site / "edit/another.html").is_file()
    assert (
        'href="../notebooks/another.html"' in (site / "examples/index.html").read_text()
    )

    new.write_text(template.format(code='raise ValueError("example failed")'))
    with pytest.raises(CalledProcessError):
        build_pages.main()
    assert page.read_text() == updated
