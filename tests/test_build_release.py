"""Rebuild PPDS 1.1.1 from a PPDS 1.0.0 download through the documented command."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ppds.paths import ROOT

SCRIPT = ROOT / "scripts" / "build_release_1_1_1.py"

SOURCE = os.environ.get("PPDS_DATA_1_0_0", "").strip()


@pytest.mark.skipif(not SOURCE, reason="set PPDS_DATA_1_0_0 to a PPDS 1.0.0 download")
def test_build_reproduces_the_distributed_1_1_1_dictionary(tmp_path):
    script = ROOT / "scripts" / "build_release_1_1_1.py"
    subprocess.run([sys.executable, script, Path(SOURCE), tmp_path], check=True, stdout=subprocess.DEVNULL)
    built = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    packaged = json.loads((ROOT / "src/ppds/release_1_1_1.json").read_text(encoding="utf-8"))
    assert built == packaged
    released = json.loads((ROOT / "src/ppds/release_1_0_0.json").read_text(encoding="utf-8"))["resources"]
    assert {name for name, spec in built["resources"].items() if spec == released.get(name)} == {"scans"}


@pytest.mark.parametrize("output", [".", "nested"])
def test_build_refuses_to_write_into_its_input(tmp_path, output):
    result = subprocess.run([sys.executable, SCRIPT, tmp_path, tmp_path / output], capture_output=True, text=True,
                            check=False)
    assert result.returncode and "separate from the 1.0.0 input" in result.stderr
