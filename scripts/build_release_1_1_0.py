"""Build PPDS 1.1.0 from a verified PPDS 1.0.0 (or 1.1.0) download.

PPDS 1.1.0 adds two pressure-map tables and a strike index for each gait
contact. Every other file is byte-identical to PPDS 1.0.0. The output
directory uses the flat Zenodo filenames and receives a new metadata.json.

    uv run --locked python scripts/build_release_1_1_0.py /path/to/1.0.0 /path/to/1.1.0

Pass --derived-only to write just the rebuilt tables and metadata.json.
"""

import argparse
import copy
import hashlib
import json
import shutil
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from reader_data import Release

VERSION = "1.1.0"
BASE = json.loads(files("ppds").joinpath("release_1_0_0.json").read_text(encoding="utf-8"))
PARQUET = {"compression": "zstd"}
ROW_GROUP_ROWS = 65536

# Readings belong to the contact of the same trial and side with start_frame <= frame < stop_frame.
CONTACT_READINGS = """
    contacts c JOIN pressure p
      ON p.trial_id = c.trial_id AND p.side = c.side
     AND p.frame >= c.start_frame AND p.frame < c.stop_frame
"""

STRIKE_QUERY = f"""
WITH first AS (
    SELECT c.trial_id, c.contact, sum(p.value::DOUBLE * p.y) / sum(p.value::DOUBLE) AS cop_y
    FROM {CONTACT_READINGS} WHERE p.frame = c.start_frame
    GROUP BY c.trial_id, c.contact
), extent AS (
    SELECT c.trial_id, c.contact, min(p.y)::DOUBLE AS y_min, max(p.y)::DOUBLE AS y_max
    FROM {CONTACT_READINGS}
    GROUP BY c.trial_id, c.contact
)
SELECT trial_id, contact, (cop_y - (y_min - 0.5)) / (y_max - y_min + 1) AS strike_index
FROM contacts JOIN first USING (trial_id, contact) JOIN extent USING (trial_id, contact)
ORDER BY trial_id, contact
"""

CONTACT_MAPS_QUERY = f"""
SELECT c.trial_id, c.contact, p.y, p.x,
       max(p.value) AS peak_pressure,
       sum(p.value::INTEGER)::DOUBLE / t.frequency_hz AS pressure_time_integral
FROM {CONTACT_READINGS} JOIN trials t ON t.trial_id = c.trial_id
GROUP BY c.trial_id, c.contact, p.y, p.x, t.frequency_hz
ORDER BY c.trial_id, c.contact, p.y, p.x
"""

STANDING_MAPS_QUERY = """
SELECT p.trial_id, p.side, p.y, p.x,
       max(p.value) AS peak_pressure,
       sum(p.value::INTEGER)::DOUBLE / t.frequency_hz AS pressure_time_integral
FROM pressure p JOIN trials t USING (trial_id)
WHERE t.condition = 'ss'
GROUP BY p.trial_id, p.side, p.y, p.x, t.frequency_hz
ORDER BY p.trial_id, p.y, p.x
"""

TRIAL_ID = "Unique release trial identifier. Join to trials.trial_id."
CONTACT = "One-based contact number in chronological order within a gait trial."
Y = "Sensor index along the long platform axis (0–352)."
X = "Sensor index across the platform (0–63)."
STRIKE = (
    "Center of pressure on the first loaded frame, measured along y from the posterior edge of "
    "the contact footprint, as a fraction of the footprint length (0 heel, 1 toe)."
)
PEAK = "Highest pressure at the sensor during the interval."
PTI = "Sum of the sensor's pressures over the interval divided by the sampling frequency."

NEW_RESOURCES = {
    "contact_maps": {
        "path": "features/contact_maps.parquet",
        "description": "Peak pressure and pressure-time integral at each sensor during each gait contact",
        "primary_key": ["trial_id", "contact", "y", "x"],
        "columns": {
            "trial_id": ("uint16", TRIAL_ID, None),
            "contact": ("uint8", CONTACT, None),
            "y": ("uint16", Y, None),
            "x": ("uint8", X, None),
            "peak_pressure": ("uint16", PEAK, "kPa"),
            "pressure_time_integral": ("double", PTI, "kPa s"),
        },
        "schema_metadata": {
            "key": "trial_id,contact,y,x; join contacts on (trial_id, contact)",
            "grid": "original platform grid, 353 rows (y) by 64 columns (x); omitted sensors carry no load",
            "interval": "zero-based [contacts.start_frame, contacts.stop_frame), the contact's foot only",
            "peak_pressure": "kPa",
            "pressure_time_integral": "kPa s; mean pressure = pressure_time_integral / contacts.contact_time",
        },
    },
    "standing_maps": {
        "path": "features/standing_maps.parquet",
        "description": "Peak pressure and pressure-time integral at each sensor during quiet standing",
        "primary_key": ["trial_id", "y", "x"],
        "columns": {
            "trial_id": ("uint16", TRIAL_ID, None),
            "side": ("string", "Anatomical foot side, L for left and R for right.", None),
            "y": ("uint16", Y, None),
            "x": ("uint8", X, None),
            "peak_pressure": ("uint16", PEAK, "kPa"),
            "pressure_time_integral": ("double", PTI, "kPa s"),
        },
        "schema_metadata": {
            "key": "trial_id,y,x; one bilateral map per quiet-standing trial",
            "grid": "original platform grid, 353 rows (y) by 64 columns (x); omitted sensors carry no load",
            "interval": "all 100 frames of the trial (10 s at 10 Hz)",
            "peak_pressure": "kPa",
            "pressure_time_integral": "kPa s; mean pressure = pressure_time_integral / trials.duration_s",
        },
    },
}

CHANGES = (
    "1.1.0 adds features/contact_maps.parquet, features/standing_maps.parquet and the "
    "contacts.strike_index column. All other files are byte-identical to 1.0.0."
)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packaged_1_1_0():
    path = files("ppds").joinpath("release_1_1_0.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def check_inputs(source):
    """Require the PPDS 1.0.0 tables; contacts may also carry the 1.1.0 strike index."""
    accepted = {name: {spec["sha256"]} for name, spec in BASE["resources"].items()}
    released = packaged_1_1_0()
    if released is not None:
        accepted["contacts"].add(released["resources"]["contacts"]["sha256"])
    for name in ("trials", "pressure", "contacts"):
        if sha256(source.resource_location(name)) not in accepted[name]:
            raise ValueError(f"Expected the PPDS 1.0.0 {name} table")


def write(table, path, metadata, row_group_size=None):
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    pq.write_table(table, path, row_group_size=row_group_size, **PARQUET)


def typed(table, columns):
    schema = pa.schema([(name, pa.type_for_alias(spec[0]), True) for name, spec in columns.items()])
    if table.column_names != schema.names:
        raise ValueError(f"Unexpected columns {table.column_names}")
    return table.cast(schema)


def resource(path, rows, entry):
    spec = {
        "path": entry["path"],
        "zenodo_filename": entry["path"].replace("/", "_"),
        "format": "parquet",
        "description": entry["description"],
        "primary_key": entry["primary_key"],
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "columns": {},
    }
    for name, (dtype, description, unit) in entry["columns"].items():
        spec["columns"][name] = {"type": dtype, "description": description}
        if unit:
            spec["columns"][name]["unit"] = unit
    return spec


def manifest(built):
    meta = copy.deepcopy(BASE)
    meta["version"] = VERSION
    meta["changes"] = CHANGES
    meta["gait_contacts"]["strike_index"] = (
        "Cavanagh and Lafortune (1980) strike index along the walking direction. The footprint is the "
        "union of the contact's loaded sensors; its posterior and anterior edges are the outer edges of "
        "the lowest and highest loaded rows."
    )
    meta["pressure_maps"] = {
        "peak_pressure": "Maximum reading at each sensor over the interval, in kPa.",
        "pressure_time_integral": (
            "Sum of each sensor's readings over the interval divided by trials.frequency_hz, in kPa s. "
            "Unloaded frames count as zero pressure."
        ),
        "mean_pressure": (
            "Not stored: pressure_time_integral divided by contacts.contact_time for a gait contact "
            "or by trials.duration_s for quiet standing."
        ),
        "gait_interval": "The contact's frames [start_frame, stop_frame) and only the contact's foot.",
        "standing_interval": "All 100 frames of the quiet-standing trial and both feet, each sensor labelled by foot.",
    }
    meta["joins"]["contact_maps"] = "Join contact_maps to contacts on (trial_id, contact) and to trials on trial_id."
    meta["joins"]["standing_maps"] = "Join standing_maps to trials on trial_id."
    resources = {}
    for name, spec in BASE["resources"].items():
        resources[name] = built.get(name, spec)
        if name == "contacts":
            for new in ("contact_maps", "standing_maps"):
                resources[new] = built[new]
    meta["resources"] = resources
    return meta


def build(data, output, *, derived_only=False):
    source = Release.local(data)
    check_inputs(source)
    output.mkdir(parents=True, exist_ok=True)
    built = {}
    with source.connect(("trials", "pressure", "contacts")) as con:
        con.execute("SET preserve_insertion_order = true")
        strike = con.sql(STRIKE_QUERY).to_arrow_table()
        contact_maps = con.sql(CONTACT_MAPS_QUERY).to_arrow_table()
        standing_maps = con.sql(STANDING_MAPS_QUERY).to_arrow_table()

    contacts_path = Path(source.resource_location("contacts"))
    contacts = pq.read_table(contacts_path)
    old_metadata = {k.decode(): v.decode() for k, v in (contacts.schema.metadata or {}).items()}
    if "strike_index" in contacts.column_names:
        contacts = contacts.drop_columns(["strike_index"])
    if not (
        strike["trial_id"].equals(contacts["trial_id"]) and strike["contact"].equals(contacts["contact"])
    ):
        raise ValueError("Every gait contact requires a loaded first frame")
    contacts = contacts.append_column(pa.field("strike_index", pa.float64()), strike["strike_index"])
    path = output / "features_contacts.parquet"
    write(contacts, path, old_metadata | {"strike_index": "fraction of footprint length from the heel edge"})
    spec = copy.deepcopy(BASE["resources"]["contacts"])
    spec.update(bytes=path.stat().st_size, sha256=sha256(path))
    spec["columns"]["strike_index"] = {"type": "double", "description": STRIKE, "unit": "1"}
    built["contacts"] = spec

    for name, table in (("contact_maps", contact_maps), ("standing_maps", standing_maps)):
        entry = NEW_RESOURCES[name]
        table = typed(table, entry["columns"])
        key = entry["primary_key"]
        if len(table.group_by(key).aggregate([])) != len(table):
            raise ValueError(f"Duplicate {name} keys")
        path = output / entry["path"].replace("/", "_")
        # Row groups sorted by trial_id let remote readers fetch one contact's map in small ranges.
        write(table, path, entry["schema_metadata"], ROW_GROUP_ROWS)
        built[name] = resource(path, len(table), entry)

    if not derived_only:
        for name, spec in BASE["resources"].items():
            if name == "contacts":
                continue
            target = output / spec["zenodo_filename"]
            src = Path(source.resource_location(name))
            if sha256(src) != spec["sha256"]:
                raise ValueError(f"Expected the PPDS 1.0.0 {name} file")
            if target.resolve() != src.resolve():
                shutil.copyfile(src, target)
    meta = manifest(built)
    (output / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="PPDS 1.0.0 or 1.1.0 directory containing metadata.json")
    parser.add_argument("output", type=Path, help="Directory for the PPDS 1.1.0 files")
    parser.add_argument("--derived-only", action="store_true", help="Write only the rebuilt tables and metadata.json")
    args = parser.parse_args()
    meta = build(args.data.resolve(), args.output.resolve(), derived_only=args.derived_only)
    print(json.dumps({name: {k: spec.get(k) for k in ("rows", "bytes")} for name, spec in meta["resources"].items()}, indent=1))
