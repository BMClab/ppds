"""Build the PPDS Pages site in _site/ from the READMEs and marimo examples."""

import ast
import os
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import markdown
from markdown.treeprocessors import Treeprocessor

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("examples/README.md")
CATALOG_URL = "https://bmclab.github.io/ppds/examples/"
SOURCE_LINK = '<a href="https://github.com/BMClab/ppds">Source on GitHub</a>'
GUIDES = (
    Path("README.md"),
    CATALOG,
    Path("data/README.md"),
    Path("docs/protocol/pressure_axes.md"),
    Path("src/ppds/README.md"),
    Path("framework/src/reader_data/README.md"),
)
STYLE = """
:root { color-scheme: light dark; font: 1rem/1.6 Inter, system-ui, sans-serif; }
body { max-width: 65rem; margin: auto; padding: 1.5rem; }
a { text-underline-offset: .15em; }
table { display: block; overflow-x: auto; border-collapse: collapse; }
td, th { text-align: left; padding: .4rem .7rem; border-bottom: 1px solid GrayText; }
pre { overflow-x: auto; padding: 1rem; border: 1px solid GrayText; border-radius: .3rem; }
"""


def is_notebook(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    constructors = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            constructors.update(
                f"{a.asname or a.name}.App" for a in node.names if a.name == "marimo"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "marimo":
            constructors.update(
                a.asname or a.name for a in node.names if a.name == "App"
            )
    return any(
        isinstance(node, ast.Call) and ast.unparse(node.func) in constructors
        for node in ast.walk(tree)
    )


class PageLinks(Treeprocessor):
    def __init__(self, source, pages, notebooks):
        super().__init__()
        self.source = source
        self.pages = pages
        self.notebooks = notebooks

    def run(self, document):
        listed = set()
        for link in document.iter("a"):
            href = link.get("href", "")
            url = urlsplit(href)
            if href == CATALOG_URL:
                target = CATALOG
            elif url.scheme or url.netloc or not url.path:
                continue
            else:
                target = (
                    (ROOT / self.source.parent / unquote(url.path))
                    .resolve()
                    .relative_to(ROOT)
                )
            if not (ROOT / target).is_file():
                raise ValueError(f"Missing link in {self.source}: {href}")
            if target in self.pages:
                path = os.path.relpath(
                    self.pages[target], self.pages[self.source].parent
                )
                link.set(
                    "href", urlunsplit(("", "", quote(path), url.query, url.fragment))
                )
                listed.add(target)
            else:
                revision = os.environ.get("GITHUB_SHA", "main")
                path = f"/BMClab/ppds/blob/{revision}/{quote(target.as_posix())}"
                link.set(
                    "href",
                    urlunsplit(("https", "github.com", path, url.query, url.fragment)),
                )

        if self.source == CATALOG:
            missing = sorted(self.notebooks - listed)
            if missing:
                ul = ET.SubElement(document, "ul")
                for source in missing:
                    href = os.path.relpath(
                        self.pages[source], self.pages[CATALOG].parent
                    )
                    ET.SubElement(
                        ET.SubElement(ul, "li"), "a", href=quote(href)
                    ).text = source.stem


def build(site):
    notebooks = {
        p.relative_to(ROOT) for p in (ROOT / "examples").glob("*.py") if is_notebook(p)
    }
    if not notebooks:
        raise ValueError("No marimo notebooks found in examples/")
    pages = {
        g: g.parent / "index.html" if g.name == "README.md" else g.with_suffix(".html")
        for g in GUIDES
    }
    pages.update(
        {n: Path("notebooks") / n.with_suffix(".html").name for n in notebooks}
    )

    for guide in GUIDES:
        md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
        md.treeprocessors.register(PageLinks(guide, pages, notebooks), "site-links", 1)
        body = md.convert((ROOT / guide).read_text())
        title = escape(md.toc_tokens[0]["name"])
        page = site / pages[guide]
        page.parent.mkdir(parents=True, exist_ok=True)
        nav = ""
        if guide != Path("README.md"):
            home = os.path.relpath(site / "index.html", page.parent)
            nav = f'<nav aria-label="Site navigation"><a href="{home}">PPDS home</a> · {SOURCE_LINK}</nav>'
        page.write_text(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><style>{STYLE}</style></head><body>{nav}<main>{body}</main></body></html>"
        )

    (site / "notebooks").mkdir()
    (site / "edit").mkdir()
    revision = os.environ.get("GITHUB_SHA", "main")
    for source in sorted(notebooks):
        page = site / pages[source]
        editor = site / "edit" / page.name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                str(ROOT / source),
                "--include-code",
                "-o",
                str(page),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "marimo", "export", "html-wasm", str(ROOT / source),
             "--mode", "edit", "--single-file", "-o", str(editor)],
            cwd=ROOT,
            check=True,
        )
        shutil.copyfile(ROOT / source, page.with_suffix(".py"))
        blob = f"BMClab/ppds/blob/{quote(revision, safe='')}/{quote(source.as_posix())}"
        nav_start = (
            '<nav aria-label="Notebook navigation" style="position:relative;z-index:1;'
            'padding:.5rem 1rem;background:Canvas;color:CanvasText;font:1rem system-ui">'
            '<a href="../index.html">PPDS home</a> · <a href="../examples/index.html">All examples</a> · '
        )
        nav = (
            f'{nav_start}{SOURCE_LINK} · '
            f'<a href="../edit/{quote(editor.name)}">Edit in browser</a> · '
            f'<a href="https://molab.marimo.io/github/{blob}">Open in molab</a> · '
            f'<a href="https://marimo.app/github.com/{blob}">Open in marimo.app</a> · '
            f'<a href="{quote(source.name)}" download>Download .py</a></nav>'
        )
        editor_nav = f'{nav_start}<a href="../notebooks/{quote(page.name)}">Saved output</a></nav>'
        for output, navigation in ((page, nav), (editor, editor_nav)):
            document = output.read_text()
            if document.count("<body>") != 1 or document.count("</head>") != 1:
                raise ValueError(f"Unexpected marimo HTML in {output}")
            if output == editor:
                # Fit the absolutely positioned editor beneath the site's navigation.
                layout = "<style>body{display:flex;flex-direction:column;height:100dvh}#root{position:relative;flex:1;min-height:0}</style>"
                document = document.replace("</head>", layout + "</head>", 1)
            output.write_text(document.replace("<body>", "<body>" + navigation, 1))
    print(f"Built {len(notebooks)} notebooks and {len(GUIDES)} guide pages.")


def main():
    with TemporaryDirectory(dir=ROOT, prefix=".pages-build-") as temporary:
        site = Path(temporary) / "site"
        build(site)
        output = ROOT / "_site"
        if output.exists():
            shutil.rmtree(output)
        site.rename(output)


if __name__ == "__main__":
    main()
