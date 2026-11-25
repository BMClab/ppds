"""Rebuild the PPDS 1.1.0 additions through the documented command."""

import json
import subprocess
import sys

from ppds.paths import ROOT


def test_build_reproduces_the_distributed_1_1_0_dictionary(data_root, tmp_path):
    script = ROOT / "scripts" / "build_release_1_1_0.py"
    subprocess.run([sys.executable, script, data_root, tmp_path, "--derived-only"], check=True,
                   stdout=subprocess.DEVNULL)
    built = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    packaged = json.loads((ROOT / "src/ppds/release_1_1_0.json").read_text(encoding="utf-8"))
    assert built == packaged
    released = json.loads((ROOT / "src/ppds/release_1_0_0.json").read_text(encoding="utf-8"))["resources"]
    changed = {name for name, spec in built["resources"].items() if spec != released.get(name)}
    assert changed == {"contacts", "contact_maps", "standing_maps"}
