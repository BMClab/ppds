"""Build the interactive, single-file browser edition of the dashboard."""

import base64
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

DASHBOARD = Path(__file__).resolve().parent


def browser_notebook():
    # Single-file WASM exports embed the notebook, but omit sibling modules.
    # Put those modules on the browser kernel's import path before its first import.
    bundle = BytesIO()
    with ZipFile(bundle, "w") as archive:
        archive.writestr(ZipInfo("dashboard_support/__init__.py"), "")
        for path in sorted((DASHBOARD / "dashboard_support").glob("*.py")):
            archive.writestr(
                ZipInfo(path.relative_to(DASHBOARD).as_posix()),
                path.read_bytes(),
                compress_type=ZIP_DEFLATED,
            )
    payload = base64.b64encode(bundle.getvalue()).decode("ascii")
    bootstrap = f"""    import base64 as _base64
    import sys as _sys
    from pathlib import Path as _Path

    _helpers = _Path("dashboard_support.zip").resolve()
    _helpers.write_bytes(_base64.b64decode("{payload}"))
    _sys.path.insert(0, str(_helpers))

"""
    notebook = (DASHBOARD / "dashboard.py").read_text(encoding="utf-8")
    first_import = notebook.index("    from dashboard_support import ")
    return notebook[:first_import] + bootstrap + notebook[first_import:]


def main():
    with TemporaryDirectory() as directory:
        notebook = Path(directory) / "dashboard.py"
        notebook.write_text(browser_notebook(), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html-wasm",
                str(notebook),
                "--mode",
                "run",
                "--single-file",
                "--no-sandbox",
                "-f",
                "-o",
                str(DASHBOARD / "dashboard.html"),
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
