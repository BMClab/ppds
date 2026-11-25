"""Reproduce the repository's quiet-standing maps through the documented CLI."""

import subprocess
import sys

import pyarrow.parquet as pq

from ppds.paths import ROOT


def test_build_standing_peak_maps(data_root, tmp_path):
    script = ROOT / "scripts" / "build_standing_peak_maps.py"
    subprocess.run([sys.executable, script, data_root, tmp_path], check=True)
    filename = "standing_peak_maps.parquet"
    built = pq.read_table(tmp_path / filename)
    committed = pq.read_table(ROOT / "data" / "features" / filename)
    assert built.equals(committed, check_metadata=True)
