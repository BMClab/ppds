"""Build PPDS 1.1.1 from a verified PPDS 1.0.0 download.

PPDS 1.1.1 corrects 1.0.0 and adds derived tables:

- 50 truncated gait contacts are excluded with their sensor readings: the 40
  contacts that load an outermost sensor column (x = 0 or 63), extending the
  1.0.0 curation rule for the outermost rows, and the ten listed in EXCLUDED.
  The affected trials keep their clocks; their reading counts, frame features,
  contact numbers and footprints follow from the remaining readings.
- features/contact_maps.parquet and features/standing_maps.parquet give each
  sensor's peak pressure and pressure-time integral per gait contact and per
  quiet-standing trial, and contacts.strike_index gives each contact's strike
  index with a 20 N onset.

Walking footprints are registered again with the 1.0.0 procedure; groups that
lost no contact keep their 1.0.0 images. The scan archive is byte-identical to
1.0.0. The output directory uses the flat Zenodo filenames and receives a new
metadata.json.

    uv run --locked python scripts/build_release_1_1_1.py /path/to/1.0.0 /path/to/1.1.1
"""

import argparse
import copy
import hashlib
import json
import shutil
from importlib.resources import files
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from reader_data import Release

from ppds import contact_geometry
from ppds.dataset import NCOLS, NROWS
from ppds.maps import STRIKE_ONSET_N

VERSION = "1.1.1"
BASE = json.loads(files("ppds").joinpath("release_1_0_0.json").read_text(encoding="utf-8"))
PARQUET = {"compression": "zstd", "compression_level": 9}  # as in 1.0.0
PRESSURE_GROUP_ROWS = 131072  # as in 1.0.0
MAP_GROUP_ROWS = 65536
PITCH = 8.469
FORCE = PITCH**2 / 1000  # N per kPa on one sensor
AREA = PITCH**2 / 100  # cm2 per sensor

# 1.0.0 (trial_id, contact) of the truncated contacts, with their keys as a cross-check.
EXCLUDED = [
    (644, 1, "s017", "rc", 9, "R", "recording started during stance"),
    (674, 1, "s017", "ws", 8, "L", "recording started during stance"),
    (345, 1, "s009", "wf", 7, "R", "heel cut off at the start of the walkway"),
    (571, 5, "s015", "wc", 7, "L", "toes cut off at the far end of the walkway"),
    (336, 6, "s009", "wc", 8, "L", "toes cut off at the far end of the walkway"),
    (350, 6, "s009", "ws", 2, "L", "toes cut off at the far end of the walkway"),
    (1246, 4, "s035", "wc", 3, "L", "toes cut off at the far end of the walkway"),
    (1562, 10, "s044", "ws", 10, "R", "toes cut off at the far end of the walkway"),
    (3030, 5, "s087", "wc", 7, "R", "toes cut off at the far end of the walkway"),
    (2068, 5, "s060", "wc", 1, "L", "toes cut off at the far end of the walkway"),
]

TRIAL_ID = "Unique release trial identifier. Join to trials.trial_id."
CONTACT = "One-based contact number in chronological order within a gait trial."
Y = f"Sensor index along the long platform axis (0–{NROWS - 1})."
X = "Sensor index across the platform (0–63)."
STRIKE = (
    "Center of pressure at onset, the first frame with at least 20 N on the foot, measured along y "
    "from the posterior edge of the contact footprint, as a fraction of the footprint length (0 heel, 1 toe)."
)
PEAK = "Highest pressure at the sensor during the interval."
PTI = "Sum of the sensor's pressures over the interval divided by the sampling frequency."
GRID_NOTE = f"original platform grid, {NROWS} rows (y) by {NCOLS} columns (x); omitted sensors carry no load"

NEW_RESOURCES = {
    "contact_maps": {
        "path": "features/contact_maps.parquet",
        "description": "Peak pressure and pressure-time integral at each loaded sensor during each gait contact",
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
            "grid": GRID_NOTE,
            "interval": "zero-based [contacts.start_frame, contacts.stop_frame), the contact's foot only",
            "peak_pressure": "kPa",
            "pressure_time_integral": "kPa s; mean pressure = pressure_time_integral / contacts.contact_time",
        },
    },
    "standing_maps": {
        "path": "features/standing_maps.parquet",
        "description": "Peak pressure and pressure-time integral at each loaded sensor during quiet standing",
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
            "grid": GRID_NOTE,
            "interval": "all 100 frames of the trial (10 s at 10 Hz)",
            "peak_pressure": "kPa",
            "pressure_time_integral": "kPa s; mean pressure = pressure_time_integral / trials.duration_s",
        },
    },
}

CHANGES = (
    "Relative to 1.0.0, 1.1.1 excludes 50 truncated gait contacts with their sensor readings "
    "(gait_contacts.excluded), so trials.n_cells, frame features, later contact numbers in the affected trials and "
    "footprint numbers in the 26 affected walking groups follow from the remaining readings, and it adds "
    "features/contact_maps.parquet, features/standing_maps.parquet and contacts.strike_index (20 N onset). "
    "Relative to 1.1.0, which kept all 1.0.0 contacts and used the first loaded frame as strike onset, 1.1.1 "
    "supersedes those tables. scans.zip is byte-identical to 1.0.0."
)
SIDE_EDGE = "loads an outermost sensor column (x = 0 or 63)"
SIDE_EDGE_COUNT = 40  # cross-check of the rule on PPDS 1.0.0

CONTACT_READINGS = """
    contacts c JOIN pressure p
      ON p.trial_id = c.trial_id AND p.side = c.side
     AND p.frame >= c.start_frame AND p.frame < c.stop_frame
"""

STRIKE_QUERY = f"""
WITH frames AS (
    SELECT c.trial_id, c.contact, p.frame, sum(p.value::DOUBLE) AS load,
           sum(p.value::DOUBLE * p.y) / sum(p.value::DOUBLE) AS cop_y
    FROM {CONTACT_READINGS}
    GROUP BY c.trial_id, c.contact, p.frame
), onset AS (
    SELECT trial_id, contact, arg_min(cop_y, frame) AS cop_y
    FROM frames WHERE load * {FORCE!r} >= {STRIKE_ONSET_N!r}
    GROUP BY trial_id, contact
), extent AS (
    SELECT c.trial_id, c.contact, min(p.y)::DOUBLE AS y_min, max(p.y)::DOUBLE AS y_max
    FROM {CONTACT_READINGS}
    GROUP BY c.trial_id, c.contact
)
SELECT trial_id, contact, (cop_y - (y_min - 0.5)) / (y_max - y_min + 1) AS strike_index
FROM contacts LEFT JOIN onset USING (trial_id, contact) LEFT JOIN extent USING (trial_id, contact)
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


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata_of(table):
    return {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}


def write(table, path, metadata, row_group_size=None):
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    pq.write_table(table, path, row_group_size=row_group_size, **PARQUET)


def typed(table, columns):
    schema = pa.schema([(name, pa.type_for_alias(spec[0]), True) for name, spec in columns.items()])
    if table.column_names != schema.names:
        raise ValueError(f"Unexpected columns {table.column_names}")
    return table.cast(schema)


def check_inputs(source):
    for name, spec in BASE["resources"].items():
        if sha256(source.resource_location(name)) != spec["sha256"]:
            raise ValueError(f"Expected the PPDS 1.0.0 {name} file")


def excluded_intervals(con):
    """Return the excluded contacts with their 1.0.0 intervals after checking every key."""
    rows = []
    for trial_id, contact, subject, condition, trial, side, _ in EXCLUDED:
        found = con.execute("""
            SELECT t.subject, t.condition, t.trial, c.side, c.start_frame, c.stop_frame
            FROM contacts c JOIN trials t USING (trial_id) WHERE c.trial_id = ? AND c.contact = ?
        """, [trial_id, contact]).fetchall()
        found = found[0] if found else None
        if found is None or found[:4] != (subject, condition, trial, side):
            raise ValueError(f"Excluded contact {trial_id}/{contact} does not match {subject} {condition} {trial} {side}")
        rows.append((trial_id, contact, side, found[4], found[5]))
    edge = con.execute(f"""
        SELECT DISTINCT c.trial_id, c.contact, t.subject, t.condition, t.trial, c.side, c.start_frame, c.stop_frame
        FROM {CONTACT_READINGS} JOIN trials t ON t.trial_id = c.trial_id
        WHERE p.x IN (0, {NCOLS - 1}) ORDER BY c.trial_id, c.contact
    """).fetchall()
    if len(edge) != SIDE_EDGE_COUNT or {(t, c) for t, c, *_ in edge} & {(t, c) for t, c, *_ in rows}:
        raise ValueError(f"Expected {SIDE_EDGE_COUNT} side-edge contacts apart from the listed ones")
    rows += [(t, c, side, start, stop) for t, c, _, _, _, side, start, stop in edge]
    side_edge = [(t, c, s, k, n, side, SIDE_EDGE) for t, c, s, k, n, side, _, _ in edge]
    return sorted(rows), sorted(EXCLUDED + side_edge)


def recompute_frames(features, pressure, affected):
    """Recompute the frame features of the affected (trial_id, frame, side) groups from the kept readings."""
    rows = features.to_pylist()
    index = {(r["trial_id"], r["frame"]): r for r in rows}
    cells = pressure.to_pandas()
    for (trial_id, frame) in sorted({(t, f) for t, f, _ in affected}):
        row = index[(trial_id, frame)]
        here = cells[(cells.trial_id == trial_id) & (cells.frame == frame)]
        for side, suffix in ((None, ""), ("L", "_left"), ("R", "_right")):
            part = here if side is None else here[here.side == side]
            load = float(part.value.sum())
            row["grf_z" + suffix] = load * FORCE
            row["contact_area" + suffix] = len(part) * AREA
            row["peak_pressure" + suffix] = int(part.value.max()) if len(part) else 0
            for axis in ("x", "y"):
                moment = float((part[axis].astype(np.int64) * part.value).sum())
                row[f"cop_{axis}{suffix}"] = PITCH * moment / load if load else None
    return pa.Table.from_pylist(rows, schema=features.schema)


def check_frame_formula(features, pressure, sample):
    """Recompute frames that the exclusions leave unchanged; they must match 1.0.0."""
    expected = recompute_frames(features, pressure, sample).to_pylist()
    stored = {(r["trial_id"], r["frame"]): r for r in features.to_pylist()}
    for row in expected:
        if (row["trial_id"], row["frame"]) not in {(t, f) for t, f, _ in sample}:
            continue
        old = stored[(row["trial_id"], row["frame"])]
        for key, value in row.items():
            if value is None or old[key] is None:
                if value is not old[key] and value != old[key]:
                    raise ValueError(f"Frame formula differs from 1.0.0 in {key}")
            elif abs(value - old[key]) > 1e-9 * max(1, abs(old[key])):
                raise ValueError(f"Frame formula differs from 1.0.0 in {key}")


def register_footprints(con):
    """Register each walking participant, speed and side with the 1.0.0 procedure."""
    cells = con.execute("""
        SELECT t.subject, t.condition, c.side, t.trial, c.trial_id, c.contact, m.y, m.x, m.peak_pressure
        FROM contacts c JOIN trials t USING (trial_id) JOIN contact_maps m USING (trial_id, contact)
        WHERE t.condition IN ('ws', 'wc', 'wf')
        ORDER BY t.subject, t.condition, c.side, t.trial, c.contact
    """).df()
    groups = []
    for (subject, condition, side), group in cells.groupby(["subject", "condition", "side"], sort=True):
        parts = []
        images, links = [], []
        for (trial_id, contact), contact_cells in group.groupby(["trial", "contact"], sort=True):
            peak = np.zeros((NROWS, NCOLS))
            peak[contact_cells.y.to_numpy(), contact_cells.x.to_numpy()] = contact_cells.peak_pressure.to_numpy()
            images.append(peak)
            links.append((int(contact_cells.trial_id.iloc[0]), int(contact)))
        template = contact_geometry.register_contact_peak(images[0])
        registered = [template] + [contact_geometry.register_contact_peak(image, template) for image in images[1:]]
        for footprint, (image, (trial_id, contact)) in enumerate(zip(registered, links)):
            y, x = np.nonzero(image > 0)
            parts.append(pa.table({
                "subject": pa.array([subject] * len(y), pa.string()),
                "condition": pa.array([condition] * len(y), pa.string()),
                "side": pa.array([side] * len(y), pa.string()),
                "footprint": pa.array(np.full(len(y), footprint), pa.uint16()),
                "y": pa.array(y, pa.uint16()),
                "x": pa.array(x, pa.uint8()),
                "value": pa.array(image[y, x], pa.float64()),
                "trial_id": pa.array(np.full(len(y), trial_id), pa.uint16()),
                "contact": pa.array(np.full(len(y), contact), pa.uint8()),
            }))
        groups.append(pa.concat_tables(parts))
    return groups


def resource(path, rows, spec):
    spec = copy.deepcopy(spec)
    spec.update(rows=rows, bytes=path.stat().st_size, sha256=sha256(path))
    return spec


def new_resource(path, rows, entry):
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


def manifest(built, excluded):
    meta = copy.deepcopy(BASE)
    meta["version"] = VERSION
    meta["changes"] = CHANGES
    meta["experiment"]["plantar_pressure"] = "Two Zebris FDM-1.5 pressure platforms placed end to end."
    meta["pressure_grid"]["junction_row"] = (
        "Rows 0-175 and 177-352 come from two 176-row platforms placed end to end. The Zebris software fills "
        "row 176 at their junction with the mean of rows 175 and 177, rounded down to the 5 kPa step; those "
        "readings count toward force, contact area and the maps."
    )
    meta["pressure_grid"]["edges"] = (
        "During curation, gait steps that loaded row 0 or row 352 were removed, and 1.1.1 removes contacts "
        "that load column 0 or 63."
    )
    meta["gait_contacts"]["strike_index"] = (
        "After Cavanagh and Lafortune (1980): the center of pressure at onset, the first frame with at least "
        "20 N on the foot, along the walking direction. The first loaded frame reaches 20 N in 43% of contacts. "
        "The footprint is the union of the contact's loaded sensors; its posterior and anterior edges are "
        "the outer edges of the lowest and highest loaded rows."
    )
    meta["gait_contacts"]["excluded"] = [
        {"trial_id": t, "contact_1_0_0": c, "subject": s, "condition": k, "trial": n, "side": d, "reason": why}
        for t, c, s, k, n, d, why in excluded
    ]
    meta["gait_contacts"]["excluded_note"] = (
        "contact_1_0_0 is the contact number in 1.0.0 and 1.1.0. In the 23 trials where an excluded contact "
        "preceded others, the later contacts move up, so their 1.0.0 (trial_id, contact) keys do not match "
        "1.1.1. Footprint numbers shift in the 26 walking groups that lost a contact; in s022 ws R and s084 wc R "
        "the excluded contact was the registration template, so every image in those groups changed. All other "
        "footprint images are identical to 1.0.0, although some carry new contact numbers. The toe and heel "
        "cases reach row 351 or row 2 with the toes or heel missing compared with the participant's other "
        "contacts. Kept contacts that reach rows 1-3 or 350-351 are at least 0.93 of the participant's median "
        "footprint length for that task and side and show a complete heel and toes. The column rule is "
        "conservative: about half of its 40 contacts load the edge column lightly (under 0.5% of their pressure) "
        "and look nearly complete. Because excluded contacts can lie between kept ones, 16 trials have two "
        "consecutive same-side contacts; check sides before pairing consecutive contacts into steps."
    )
    meta["gait_contacts"]["boundaries"] = (
        "A contact runs from the first to the last frame with any reading on that foot, at the platform's "
        "detection limit (5 kPa on one sensor, about 0.36 N). A 20 N force threshold, common with force plates, "
        "would shorten a median contact by 10-30 ms (2-4%) and remove about 0.05% of its impulse; the per-foot "
        "force in pressure_features gives thresholded times."
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
        "sparsity": "Each row is one sensor loaded during the interval. Omitted sensors carry no load.",
    }
    meta["joins"]["contact_maps"] = "Join contact_maps to contacts on (trial_id, contact) and to trials on trial_id."
    meta["joins"]["standing_maps"] = "Join standing_maps to trials on trial_id."
    resources = {}
    for name, spec in BASE["resources"].items():
        resources[name] = built.get(name, spec)
        if name == "contacts":
            resources["contact_maps"] = built["contact_maps"]
            resources["standing_maps"] = built["standing_maps"]
    meta["resources"] = resources
    return meta


def build(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Write PPDS 1.1.1 to a directory separate from the 1.0.0 input")
    source = Release.local(data)
    check_inputs(source)
    output.mkdir(parents=True, exist_ok=True)
    built = {}
    paths = {name: Path(source.resource_location(name)) for name in BASE["resources"]}

    with source.connect(("trials", "pressure", "contacts")) as con:
        excluded, listed = excluded_intervals(con)
        con.execute("CREATE TEMP TABLE excluded (trial_id USMALLINT, contact UTINYINT, side VARCHAR, "
                    "start_frame USMALLINT, stop_frame USMALLINT)")
        con.executemany("INSERT INTO excluded VALUES (?, ?, ?, ?, ?)", excluded)
        if con.execute("SELECT count(*) FROM pressure WHERE y >= ?", [NROWS]).fetchall()[0][0]:
            raise ValueError(f"Readings beyond sensor row {NROWS - 1}")
        pressure = con.sql("""
            SELECT p.* FROM pressure p ANTI JOIN excluded e
              ON p.trial_id = e.trial_id AND p.side = e.side
             AND p.frame >= e.start_frame AND p.frame < e.stop_frame
            ORDER BY p.trial_id, p.frame, p.y, p.x
        """).to_arrow_table()
        removed = con.sql("""
            SELECT p.trial_id, p.frame, p.side FROM pressure p SEMI JOIN excluded e
              ON p.trial_id = e.trial_id AND p.side = e.side
             AND p.frame >= e.start_frame AND p.frame < e.stop_frame
        """).fetchall()
        affected_trials = sorted({t for t, _, _, _, _ in excluded})
        affected_cells = con.sql(f"""
            SELECT * FROM pressure WHERE trial_id IN ({", ".join(map(str, affected_trials))})
            ORDER BY trial_id, frame, y, x
        """).to_arrow_table()

    old_pressure = pq.read_schema(paths["pressure"])
    pressure = pressure.cast(old_pressure.remove_metadata())
    metadata = {k.decode(): v.decode() for k, v in old_pressure.metadata.items()} | {"grid": f"{NROWS},{NCOLS}"}
    path = output / "pressure.parquet"
    write(pressure, path, metadata, PRESSURE_GROUP_ROWS)
    built["pressure"] = resource(path, len(pressure), BASE["resources"]["pressure"])

    trials = pq.read_table(paths["trials"])
    counts = dict(duckdb.sql("SELECT trial_id, count(*) FROM pressure GROUP BY trial_id").fetchall())
    n_cells = pa.array([counts.get(t, 0) for t in trials["trial_id"].to_pylist()], trials.schema.field("n_cells").type)
    trials = trials.set_column(trials.schema.get_field_index("n_cells"), trials.schema.field("n_cells"), n_cells)
    path = output / "trials.parquet"
    write(trials, path, metadata_of(trials))
    built["trials"] = resource(path, len(trials), BASE["resources"]["trials"])

    # Recompute only the frames that lost readings; check the formula on frames that kept theirs.
    kept_cells = duckdb.sql(f"""
        SELECT * FROM pressure WHERE trial_id IN ({", ".join(map(str, affected_trials))})
    """).to_arrow_table()
    features_file = pq.ParquetFile(paths["pressure_features"])
    touched = {(t, f) for t, f, _ in removed}
    sample = sorted({(t, f, None) for t, f in duckdb.sql("""
        SELECT DISTINCT trial_id, frame FROM affected_cells
    """).fetchall() if (t, f) not in touched})[:200]
    path = output / "features_pressure_features.parquet"
    feature_metadata = metadata_of(features_file.schema_arrow.empty_table())
    with pq.ParquetWriter(path, features_file.schema_arrow.remove_metadata().with_metadata(
            {k.encode(): v.encode() for k, v in feature_metadata.items()}), **PARQUET) as writer:
        for group in range(features_file.num_row_groups):
            table = features_file.read_row_group(group)
            ids = set(table["trial_id"].to_pylist())
            if ids & {t for t, _, _ in sample}:
                check_frame_formula(table, affected_cells, [s for s in sample if s[0] in ids])
            if ids & set(affected_trials):
                table = recompute_frames(table, kept_cells, [(t, f, s) for t, f, s in removed if t in ids])
            writer.write_table(table)
    built["pressure_features"] = resource(path, features_file.metadata.num_rows, BASE["resources"]["pressure_features"])

    old_contacts = pq.read_table(paths["contacts"])
    excluded_keys = {(t, c) for t, c, *_ in excluded}
    mask = pa.array([(t, c) not in excluded_keys for t, c in zip(
        old_contacts["trial_id"].to_pylist(), old_contacts["contact"].to_pylist())])
    contacts = old_contacts.filter(mask)
    renumbered = duckdb.sql("""
        SELECT row_number() OVER (PARTITION BY trial_id ORDER BY start_frame) AS contact FROM contacts
        ORDER BY trial_id, start_frame
    """).to_arrow_table()["contact"].cast(pa.uint8())
    contacts = contacts.set_column(contacts.schema.get_field_index("contact"), contacts.schema.field("contact"), renumbered)

    con = duckdb.connect()
    con.register("trials", trials)
    con.register("pressure", pressure)
    con.register("contacts", contacts)
    strike = con.sql(STRIKE_QUERY).to_arrow_table()
    if strike["strike_index"].null_count or not (
        strike["trial_id"].equals(contacts["trial_id"]) and strike["contact"].equals(contacts["contact"])
    ):
        raise ValueError("Every gait contact requires an onset frame")
    contacts = contacts.append_column(pa.field("strike_index", pa.float64()), strike["strike_index"])
    path = output / "features_contacts.parquet"
    write(contacts, path, metadata_of(old_contacts) | {"strike_index": "fraction of footprint length from the heel edge, 20 N onset"})
    spec = resource(path, len(contacts), BASE["resources"]["contacts"])
    spec["columns"]["strike_index"] = {"type": "double", "description": STRIKE, "unit": "1"}
    built["contacts"] = spec

    con.register("contacts", contacts)
    for name, query in (("contact_maps", CONTACT_MAPS_QUERY), ("standing_maps", STANDING_MAPS_QUERY)):
        entry = NEW_RESOURCES[name]
        table = typed(con.sql(query).to_arrow_table(), entry["columns"])
        if len(table.group_by(entry["primary_key"]).aggregate([])) != len(table):
            raise ValueError(f"Duplicate {name} keys")
        path = output / entry["path"].replace("/", "_")
        # Row groups sorted by trial_id let remote readers fetch one map in small ranges.
        write(table, path, entry["schema_metadata"], MAP_GROUP_ROWS)
        built[name] = new_resource(path, len(table), entry)
        con.register(name, table)

    old_footprints = pq.ParquetFile(paths["footprints"])
    meta = json.loads(old_footprints.schema_arrow.metadata[b"ppds"])
    meta["version"] = VERSION
    schema = old_footprints.schema_arrow.remove_metadata().with_metadata({b"ppds": json.dumps(meta).encode()})
    path = output / "features_footprints.parquet"
    rows = 0
    with pq.ParquetWriter(path, schema, **PARQUET) as writer:
        # One row group per participant, speed and side, as in 1.0.0.
        for group in register_footprints(con):
            group = group.sort_by([("footprint", "ascending"), ("y", "ascending"), ("x", "ascending")]).cast(schema)
            writer.write_table(group)
            rows += len(group)
    built["footprints"] = resource(path, rows, BASE["resources"]["footprints"])

    shutil.copyfile(paths["scans"], output / BASE["resources"]["scans"]["zenodo_filename"])
    release = manifest(built, listed)
    (output / "metadata.json").write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="PPDS 1.0.0 directory containing metadata.json")
    parser.add_argument("output", type=Path, help="Directory for the PPDS 1.1.1 files")
    args = parser.parse_args()
    meta = build(args.data.resolve(), args.output.resolve())
    print(json.dumps({name: {k: spec.get(k) for k in ("rows", "bytes")} for name, spec in meta["resources"].items()}, indent=1))
