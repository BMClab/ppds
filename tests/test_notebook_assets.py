"""Check the wheels a notebook installs against this checkout and its Git pins."""

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement

from ppds import _browser_sample

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "notebook-assets"
RAW = "https://raw.githubusercontent.com/BMClab/ppds/"


def wheel_contents(path):
    with ZipFile(path) as wheel:
        contents = {}
        for name in wheel.namelist():
            if name.endswith(("/", ".dist-info/RECORD")):
                continue
            data = wheel.read(name)
            if name.endswith(".dist-info/METADATA"):
                data = data.partition(b"\n\n")[0]  # The project page can change independently.
            elif name.endswith(".dist-info/WHEEL"):
                data = b"\n".join(line for line in data.splitlines() if not line.startswith(b"Generator:"))
            contents[name] = data
        return contents


def test_stored_wheels_match_fresh_build(tmp_path):
    subprocess.run(["uv", "build", "--all-packages", "--wheel", "--out-dir", str(tmp_path)],
                   cwd=ROOT, check=True)
    built = {p.name for p in tmp_path.glob("*.whl")}
    assert built == {p.name for p in ASSETS.glob("*.whl")}
    for name in sorted(built):
        expected = wheel_contents(tmp_path / name)
        stored = wheel_contents(ASSETS / name)
        assert stored.keys() == expected.keys(), name
        for member, payload in expected.items():
            assert stored[member] == payload, f"Rebuild {name}: {member} changed"


def test_example_notebooks_pin_the_stored_wheels(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts.build_pages import is_notebook

    expected = {p.name.split("-")[0].replace("_", "-"): p.name for p in ASSETS.glob("*.whl")}
    common = None
    for notebook in sorted((ROOT / "examples").glob("*.py")):
        if not is_notebook(notebook):
            continue
        block = re.search(r"^# /// script\n(.*?)^# ///$", notebook.read_text(), re.MULTILINE | re.DOTALL)
        assert block, f"{notebook.name} needs script metadata"
        metadata = tomllib.loads("\n".join(line.removeprefix("#").lstrip() for line in block[1].splitlines()))
        requirements = [Requirement(dep) for dep in metadata["dependencies"]]
        pins = {}
        for req in requirements:
            name = req.name.lower().replace("_", "-")
            if name in expected:
                assert name not in pins and req.url and not req.marker, (notebook.name, req)
                pins[name] = req.url
        assert pins.keys() == expected.keys(), notebook.name
        if common is None:
            common = pins
        assert pins == common, notebook.name
    assert common
    for name, url in common.items():
        check_git_pin(url, ASSETS / expected[name])


def check_git_pin(url, stored):
    match = re.fullmatch(re.escape(RAW) + r"([0-9a-f]{40})/((?:notebook-assets/)?" +
                         re.escape(stored.name) + ")", url)
    assert match, f"Expected an immutable URL for {stored.name}: {url}"
    revision, path = match.groups()
    if not (ROOT / ".git").exists():
        pytest.skip("Git history is required to verify immutable asset URLs")
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT, check=True)
    pinned = subprocess.check_output(["git", "rev-parse", "--verify", f"{revision}:{path}"], cwd=ROOT)
    local = subprocess.check_output(["git", "hash-object", str(stored)], cwd=ROOT)
    assert pinned == local, f"Update the pin for {stored.name}: {url} serves different bytes"


def test_pressure_sample_matches_digest_and_git_pin():
    sample = ASSETS / "pressure-sample.zip"
    with sample.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == _browser_sample.SHA256
    check_git_pin(_browser_sample.SOURCE, sample)
