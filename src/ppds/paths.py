"""Locate a downloaded release in a checkout or a wheel installation."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Leave the wheel default relative so each local source selection follows the working directory.
_DEFAULT_DATA = ROOT / "data" if Path(__file__).resolve().parent.parent.name == "src" else Path("data")


def local_data_dir():
    """Resolve PPDS_DATA, falling back to data/ in the checkout or current working directory."""
    configured = os.environ.get("PPDS_DATA", "")
    path = Path(configured) if configured.strip() else _DEFAULT_DATA
    return path.expanduser().resolve()
