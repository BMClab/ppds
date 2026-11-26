"""Build the PPDS Pages site in _site/ from the READMEs and marimo examples."""

import ast
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import markdown
from markdown.treeprocessors import Treeprocessor

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("examples/README.md")
CATALOG_URL = "https://bmclab.github.io/ppds/examples/"
# Saved output of the manuscript notebook, committed from the run behind the article.
PAPER_PAGE = Path("paper/data_descriptor.html")
DASHBOARD_PAGE = Path("examples/dashboard/dashboard.html")
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


class ExportTags(HTMLParser):
    """Locate the end of the head and the opening body in the marimo export."""

    def __init__(self):
        super().__init__()
        self.head_end = self.body_start = None

    def handle_endtag(self, tag):
        if tag == "head" and self.head_end is None:
            self.head_end = self.getpos()

    def handle_starttag(self, tag, attrs):
        if tag == "body" and self.body_start is None:
            self.body_start = self.getpos()
            self.body_tag = self.get_starttag_text()


def add_navigation(output, navigation, style=""):
    document = output.read_text(encoding="utf-8")
    tags = ExportTags()
    offsets = [0]
    for line in StringIO(document):
        tags.feed(line)
        if tags.body_start is not None:
            break
        offsets.append(offsets[-1] + len(line))
    if tags.head_end is None or tags.body_start is None or tags.head_end >= tags.body_start:
        raise ValueError(f"Unexpected marimo HTML in {output}")
    line, col = tags.head_end
    head = offsets[line - 1] + col
    line, col = tags.body_start
    body = offsets[line - 1] + col + len(tags.body_tag)
    output.write_text(document[:head] + style + document[head:body] + navigation + document[body:],
                      encoding="utf-8")


def relative_url(target, directory):
    return quote(target.relative_to(directory, walk_up=True).as_posix())


class PageLinks(Treeprocessor):
    def __init__(self, source, pages, notebooks, revision):
        super().__init__()
        self.source = source
        self.pages = pages
        self.notebooks = notebooks
        self.revision = revision

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
                path = relative_url(
                    self.pages[target], self.pages[self.source].parent
                )
                link.set(
                    "href", urlunsplit(("", "", path, url.query, url.fragment))
                )
                listed.add(target)
            else:
                path = f"/BMClab/ppds/blob/{self.revision}/{quote(target.as_posix())}"
                link.set(
                    "href",
                    urlunsplit(("https", "github.com", path, url.query, url.fragment)),
                )

        if self.source == CATALOG:
            missing = sorted(self.notebooks - listed)
            if missing:
                ul = ET.SubElement(document, "ul")
                for source in missing:
                    href = relative_url(
                        self.pages[source], self.pages[CATALOG].parent
                    )
                    ET.SubElement(
                        ET.SubElement(ul, "li"), "a", href=href
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
    if (ROOT / DASHBOARD_PAGE).is_file():
        output = site / DASHBOARD_PAGE
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / DASHBOARD_PAGE, output)
        pages[DASHBOARD_PAGE] = DASHBOARD_PAGE

    revision = quote(os.environ.get("GITHUB_SHA") or "main", safe="")
    for guide in GUIDES:
        md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
        md.treeprocessors.register(PageLinks(guide, pages, notebooks, revision), "site-links", 1)
        body = md.convert((ROOT / guide).read_text(encoding="utf-8"))
        title = md.toc_tokens[0]["name"]  # The toc extension already escapes HTML.
        page = site / pages[guide]
        page.parent.mkdir(parents=True, exist_ok=True)
        nav = ""
        if guide != Path("README.md"):
            home = relative_url(site / "index.html", page.parent)
            nav = f'<nav aria-label="Site navigation"><a href="{home}">PPDS home</a> · {SOURCE_LINK}</nav>'
        page.write_text(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><style>{STYLE}</style></head><body>{nav}<main>{body}</main></body></html>",
            encoding="utf-8",
        )

    (site / "notebooks").mkdir()
    (site / "edit").mkdir()
    nav_start = (
        '<nav aria-label="Notebook navigation" style="position:relative;z-index:1;'
        'padding:.5rem 1rem;background:Canvas;color:CanvasText;font:1rem system-ui">'
        '<a href="../index.html">PPDS home</a> · '
    )
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
        blob = f"BMClab/ppds/blob/{revision}/{quote(source.as_posix())}"
        nav = (
            f'{nav_start}<a href="../examples/index.html">All examples</a> · {SOURCE_LINK} · '
            f'<a href="../edit/{quote(editor.name)}">Edit in browser</a> · '
            f'<a href="https://molab.marimo.io/github/{blob}">Open in molab</a> · '
            f'<a href="https://marimo.app/github.com/{blob}">Open in marimo.app</a> · '
            f'<a href="{quote(source.name)}" download>Download .py</a></nav>'
        )
        editor_nav = (
            f'{nav_start}<a href="../examples/index.html">All examples</a> · '
            f'<a href="../notebooks/{quote(page.name)}">Saved output</a></nav>'
        )
        add_navigation(page, nav)
        # Fit the absolutely positioned editor beneath the site's navigation.
        layout = "<style>body{display:flex;flex-direction:column;height:100dvh}#root{position:relative;flex:1;min-height:0}</style>"
        add_navigation(editor, editor_nav, layout)
    if (ROOT / PAPER_PAGE).is_file():
        # Publish the committed copy instead of re-running it, so the page keeps the article's run record.
        page = site / PAPER_PAGE
        source = PAPER_PAGE.with_suffix(".py")
        page.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / PAPER_PAGE, page)
        shutil.copyfile(ROOT / source, page.with_suffix(".py"))
        add_navigation(
            page,
            f'{nav_start}{SOURCE_LINK} · <a href="{quote(source.name)}" download>Download .py</a></nav>',
        )
        print(f"Published the saved {PAPER_PAGE}.")
    else:
        print(f"No saved {PAPER_PAGE}; the site omits the manuscript notebook.")
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
