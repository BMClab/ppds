"""Build the browser pressure sample from a PPDS 1.1.0 download with verified Parquet tables."""

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pyarrow as pa
import pyarrow.parquet as pq
from reader_data import Release

SUBJECTS = ("s001", "s005", "s055", "s058")


def build(data, output):
    source = Release.local(data)
    pinned = json.loads(files("ppds").joinpath("release_1_1_0.json").read_bytes())
    if source.metadata != pinned:
        raise ValueError("Expected the PPDS 1.1.0 data dictionary")
    for name, spec in pinned["resources"].items():
        if spec["format"] == "parquet":
            with open(source.resource_location(name), "rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch for {name}")
    trials = pq.read_table(source.resource_location("trials"), filters=[("subject", "in", SUBJECTS)])
    ids = trials["trial_id"].to_pylist()
    manifest = {"dataset": "PPDS browser sample", "source_doi": "10.5281/zenodo.24544192",
                "subjects": SUBJECTS, "subject_columns": pinned["subject_columns"], "resources": {}}
    with ZipFile(output, "w") as archive:
        for name, spec in pinned["resources"].items():
            if spec["format"] != "parquet":
                continue
            table = trials if name == "trials" else pq.read_table(
                source.resource_location(name), filters=[("trial_id", "in", ids)]
            )
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression="zstd")
            archive.writestr(ZipInfo(spec["path"]), buffer.getvalue().to_pybytes())
            manifest["resources"][name] = {"path": spec["path"], "rows": table.num_rows}
        archive.writestr(ZipInfo("metadata.json"), json.dumps(manifest, indent=2))
    print(f"{output}: {len(SUBJECTS)} participants, {len(ids)} trials, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.data, args.output)
