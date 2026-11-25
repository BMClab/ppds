"""Rebuild the committed quiet-standing peak-pressure maps from PPDS 1.0.0."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from reader_data import Release

from ppds.release_layout import REFERENCE

PEAK_QUERY = """SELECT t.trial_id, t.subject, p.side, p.y, p.x, max(p.value) AS peak_pressure
FROM pressure p JOIN trials t USING (trial_id)
WHERE t.condition = 'ss'
GROUP BY t.trial_id, t.subject, p.side, p.y, p.x
ORDER BY t.trial_id, p.side, p.y, p.x"""


def build(data, output_directory):
    """Write one sparse bilateral map per trial and return the Parquet path."""
    source = Release.local(data)
    inputs = ("trials", "pressure")
    for name in inputs:
        with source.open_resource(name) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != REFERENCE["resources"][name]["sha256"]:
            raise ValueError(f"Expected the PPDS 1.0.0 {name} table")

    with source.connect(inputs) as con:
        peaks = con.sql(PEAK_QUERY).to_arrow_table()
    peaks = peaks.replace_schema_metadata({
        "source_doi": "10.5281/zenodo.24432612",
        "units": "kPa",
        "grid": "353 rows (y) x 64 columns (x), zero based, original platform coordinates, omitted cells are zero",
        "description": (
            "Quiet standing. Per-sensor maximum over all 100 released frames (10 Hz). "
            "One bilateral map per trial, with anatomical side L or R on each loaded sensor."
        ),
        "source_query": PEAK_QUERY,
    })

    output = Path(output_directory) / "standing_peak_maps.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        pq.write_table(peaks, staged, compression="zstd")
        staged.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="PPDS 1.0.0 directory containing metadata.json")
    parser.add_argument("output_directory", type=Path, help="Directory for standing_peak_maps.parquet")
    args = parser.parse_args()
    print(build(args.data, args.output_directory))
