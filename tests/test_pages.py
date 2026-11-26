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
    editor_html = editor.read_text()
    assert "marimo" in editor_html
    assert 'href="../index.html"' in editor_html
    assert 'href="../examples/index.html"' in editor_html
    assert 'href="../notebooks/demo.html"' in editor_html
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


def test_pages_preserves_exported_html_and_uses_main_for_an_empty_revision(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts import build_pages

    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "demo.py").write_text("import marimo\napp = marimo.App()\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Pressão · PPDS\n\n[License](LICENSE)", encoding="utf-8")
    (tmp_path / "LICENSE").touch()
    monkeypatch.setattr(build_pages, "ROOT", tmp_path)
    monkeypatch.setattr(build_pages, "GUIDES", [Path("README.md")])
    monkeypatch.setenv("GITHUB_SHA", "")
    head = '<!-- <body> --><script>const literal = "</head><body>";</script>'
    body = '<BODY\n data-title="≤ or >">'
    payload = '<script>const literal = "<body>";</script>São Paulo'

    def export(args, **kwargs):
        Path(args[-1]).write_text(f"<html><head>{head}</head>{body}{payload}</body></html>", encoding="utf-8")

    monkeypatch.setattr(build_pages.subprocess, "run", export)
    site = tmp_path / "site"
    build_pages.build(site)
    guide = (site / "index.html").read_text(encoding="utf-8")
    assert "Pressão" in guide and "/blob/main/LICENSE" in guide
    for folder in ("edit", "notebooks"):
        html = (site / folder / "demo.html").read_text(encoding="utf-8")
        assert head in html and body + "<nav" in html and "</nav>" + payload in html
    assert "/blob/main/examples/demo.py" in html


def test_windows_paths_generate_portable_page_urls(monkeypatch):
    from pathlib import PureWindowsPath

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.build_pages import relative_url

    site = PureWindowsPath("C:/Documents/PPDS")
    assert relative_url(site / "notebooks/ação 1.html", site / "examples") == "../notebooks/a%C3%A7%C3%A3o%201.html"


def test_pages_publishes_the_saved_paper_page_without_running_it(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts import build_pages

    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "demo.py").write_text("import marimo\napp = marimo.App()\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# PPDS", encoding="utf-8")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "data_descriptor.py").write_text("import marimo\napp = marimo.App()\n", encoding="utf-8")
    saved = "<html><head></head><body>Run started on 2026-11-26 01:16 UTC</body></html>"
    (paper / "data_descriptor.html").write_text(saved, encoding="utf-8")
    monkeypatch.setattr(build_pages, "ROOT", tmp_path)
    monkeypatch.setattr(build_pages, "GUIDES", [Path("README.md")])
    exported = []

    def export(args, **kwargs):
        exported.append(Path(args[-1]))
        Path(args[-1]).write_text("<html><head></head><body></body></html>", encoding="utf-8")

    monkeypatch.setattr(build_pages.subprocess, "run", export)
    site = tmp_path / "site"
    build_pages.build(site)
    assert all(path.parent.name != "paper" for path in exported)
    html = (site / "paper/data_descriptor.html").read_text(encoding="utf-8")
    assert "Run started on 2026-11-26 01:16 UTC" in html and 'href="../index.html"' in html
    assert 'href="data_descriptor.py" download' in html
    assert (site / "paper/data_descriptor.py").read_text() == (paper / "data_descriptor.py").read_text()
