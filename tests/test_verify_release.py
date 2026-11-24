"""Exercise the download verifier with small, internally checksummed releases."""

import hashlib
import json
import zipfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ppds.paths import ROOT
from ppds.release_layout import SUBJECT_COLUMNS, TABLE_SCHEMAS

SUBJECTS = list(SUBJECT_COLUMNS)
SIDE_COVERAGE = {"wc": {"frames": 4, "unresolved_per_foot_frames": 0}}


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from scripts import verify_release

    # Two gait trials, four frames, and two loaded cells in this test release.
    monkeypatch.setattr(
        verify_release, "EXPECTED_COVERAGE", (2, 2, 0, 4, 2, 2), raising=False
    )
    return verify_release.verify


def write_manifest(root, resources, **overrides):
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "resources": resources,
                "subject_columns": SUBJECTS,
                "pressure_features": {"side_coverage": SIDE_COVERAGE},
                **overrides,
            }
        )
    )


def replace_values(table, column, values):
    field = table.schema.field(column)
    return table.set_column(
        table.schema.get_field_index(column), field, [pa.array(values, type=field.type)]
    )


def file_info(path):
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def resources(tmp_path):
    trials = {
        "trial_id": [0, 1],
        "subject": ["s001", "s002"],
        "condition": ["wc"] * 2,
        "trial": [1, 1],
        "n_frames": [2, 2],
        "frequency_hz": [100, 100],
        "duration_s": [0.02, 0.02],
        "n_cells": [1, 1],
    }
    for name in SUBJECTS[1:]:
        trials[name] = [None, None]
    for name in ("TrialsWS", "TrialsWC", "TrialsWF", "TrialsRC", "TrialsStanding"):
        trials[name] = [int(name == "TrialsWC")] * 2
    pressure = {
        "trial_id": [0, 1],
        "frame": [0, 1],
        "y": [0, 0],
        "x": [0, 0],
        "value": [1, 1],
        "side": ["L", "R"],
    }
    frames = {"trial_id": [0, 0, 1, 1], "frame": [0, 1, 0, 1]}
    for side in ("", "_left", "_right"):
        loaded = [side != "_right", False, False, side != "_left"]
        frames["grf_z" + side] = [8.469**2 / 1000 if active else 0 for active in loaded]
        for axis in ("x", "y"):
            frames["cop_" + axis + side] = [
                0.0 if active else None for active in loaded
            ]
        frames["contact_area" + side] = [
            8.469**2 / 100 if active else 0 for active in loaded
        ]
        frames["peak_pressure" + side] = [1 if active else 0 for active in loaded]
    contacts = {
        "trial_id": [0, 1],
        "contact": [1, 1],
        "side": ["L", "R"],
        "start_frame": [0, 1],
        "stop_frame": [1, 2],
        "contact_time": [0.01] * 2,
        "peak_pressure": [1] * 2,
        "peak_grf_z": [8.469**2 / 1000] * 2,
        "vertical_impulse": [8.469**2 / 100000] * 2,
        "max_contact_area": [8.469**2 / 100] * 2,
        "contact_y_min": [0.0] * 2,
        "contact_y_max": [0.0] * 2,
    }
    footprints = {
        "subject": ["s001", "s002"],
        "condition": ["wc"] * 2,
        "side": ["L", "R"],
        "footprint": [0] * 2,
        "y": [0] * 2,
        "x": [0] * 2,
        "value": [1.0] * 2,
        "trial_id": [0, 1],
        "contact": [1, 1],
    }
    resources = {}
    for name, table in (
        ("trials", trials),
        ("pressure", pressure),
        ("pressure_features", frames),
        ("contacts", contacts),
        ("footprints", footprints),
    ):
        relative = (
            ("" if name in {"trials", "pressure"} else "features/") + name + ".parquet"
        )
        path = tmp_path / relative.replace("/", "_")
        # Nullable fields allow fixtures to represent individual corrupt readings.
        schema = pa.schema([(f.name, f.type) for f in TABLE_SCHEMAS[name]])
        table = pa.table(table, schema=schema)
        pq.write_table(table, path)
        resources[name] = dict(
            path=relative,
            zenodo_filename=path.name,
            format="parquet",
            columns={f.name: {"type": str(f.type)} for f in schema},
            rows=len(table),
            **file_info(path),
        )
    archive = tmp_path / "scans.zip"
    with zipfile.ZipFile(archive, "w") as scan_zip:
        for number in range(1, 96):
            for side in "lr":
                scan_zip.writestr(
                    f"scans/s{number:03d}{side}_foot.stl",
                    bytes(80) + (1).to_bytes(4, "little") + bytes(50),
                )
    resources["scans"] = dict(
        path=archive.name,
        zenodo_filename=archive.name,
        format="zip",
        members=190,
        **file_info(archive),
    )
    write_manifest(tmp_path, resources)
    return resources


def test_verifies_custom_flat_filename(tmp_path, resources, verifier):
    resource = resources["pressure_features"]
    (tmp_path / resource["zenodo_filename"]).rename(tmp_path / "measurements.parquet")
    resource["zenodo_filename"] = "measurements.parquet"
    write_manifest(tmp_path, resources)
    assert verifier(tmp_path)["verified"]


@pytest.mark.parametrize("override", [False, True])
def test_release_commands_select_and_report_directory(
    tmp_path, resources, verifier, monkeypatch, capsys, override
):
    from scripts import verify_release

    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("PPDS_DATA", str(tmp_path / "unused" if override else tmp_path))
    args = ["--data", tmp_path.name] if override else []
    verify_release.main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] and report["data_dir"] == str(tmp_path)


@pytest.mark.parametrize("name", ["scans", "contacts", "footprints"])
def test_required_resource_missing_from_manifest_and_directory(
    tmp_path, resources, verifier, name
):
    (tmp_path / resources.pop(name)["zenodo_filename"]).unlink()
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="resource inventory"):
        verifier(tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "features/grf.parquet",
        "features_grf_cycle.parquet",
        "spatiotemporal.parquet",
        "standing.parquet",
        "stps_coding.txt",
    ],
)
def test_verifier_rejects_unlisted_vendor_files(tmp_path, resources, verifier, path):
    extra = tmp_path / path
    extra.parent.mkdir(exist_ok=True)
    extra.touch()
    with pytest.raises(ValueError, match="Excluded file"):
        verifier(tmp_path)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "crc"])
def test_inconsistent_scan_archive(tmp_path, resources, verifier, damage):
    path = tmp_path / "scans.zip"
    with zipfile.ZipFile(path) as source:
        meshes = [(entry.filename, source.read(entry)) for entry in source.infolist()]
    if damage == "missing":
        meshes.pop()
    elif damage == "duplicate":
        meshes[-1] = meshes[0]
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in meshes:
            if damage == "duplicate" and name in archive.namelist():
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, payload)
            else:
                archive.writestr(name, payload)
    if damage == "crc":
        with zipfile.ZipFile(path) as archive:
            entry = archive.infolist()[0]
        # A stored member's bytes follow its local header and filename.
        with path.open("r+b") as stream:
            stream.seek(entry.header_offset + 30 + len(entry.filename.encode()))
            stream.write(b"X")
    resources["scans"].update(file_info(path))
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="scan archive"):
        verifier(tmp_path)


@pytest.mark.parametrize(
    ("name", "column", "values", "message"),
    [
        ("pressure_features", "trial_id", [0, 0, 1, 999], "frame series"),
        ("pressure", "trial_id", [0, 999], "sensor"),
        ("trials", "n_cells", [0, 2], "sensor"),
        ("trials", "duration_s", [0.2, 0.02], "duration"),
        ("trials", "trial", [0, 1], "trial number"),
        ("trials", "trial", [None, 1], "trial number"),
        ("trials", "subject", ["s096", "s002"], "participant ID"),
        ("trials", "TrialsWC", [2, 1], "availability"),
        ("trials", "TrialsStanding", [1, 0], "availability"),
        ("footprints", "value", [1.01, 1.0], "footprint pixels or contact links"),
        ("pressure", "value", [0, 1], "sensor"),
        ("pressure", "x", [64, 0], "sensor"),
        ("pressure", "frame", [2, 1], "sensor"),
        ("pressure", "side", ["L", None], "sensor"),
        ("pressure", "side", ["?", "R"], "sensor"),
        ("pressure", "side", ["R", "L"], "feature mismatch"),
        (
            "pressure_features",
            "cop_x_right",
            [None, None, None, 10.0],
            "feature mismatch",
        ),
        ("pressure_features", "grf_z_left", [None, 0.0, 0.0, 0.0], "feature mismatch"),
        ("pressure_features", "contact_area", [0.0] * 4, "feature mismatch"),
        (
            "pressure_features",
            "peak_pressure_left",
            [1, None, 0, 0],
            "feature mismatch",
        ),
        ("contacts", "contact", [0, 1], "contact numbering"),
        ("contacts", "start_frame", [0, 0], "contact pressure, interval"),
        ("contacts", "side", ["L", "L"], "contact pressure, interval"),
        ("contacts", "peak_pressure", [1, 2], "contact pressure, interval"),
        ("contacts", "peak_grf_z", [1.0, 2.0], "contact pressure, interval"),
        ("contacts", "vertical_impulse", [None, 2.0], "contact pressure, interval"),
        ("contacts", "max_contact_area", [1.0, 2.0], "contact pressure, interval"),
        ("contacts", "contact_y_min", [1.0, 0.0], "contact pressure, interval"),
        ("contacts", "contact_y_max", [0.0, 1.0], "contact pressure, interval"),
        ("contacts", "contact_time", [0.05, 0.01], "contact pressure, interval"),
        ("contacts", "trial_id", [999, 1], "contact numbering"),
        ("footprints", "trial_id", [1, 0], "footprint pixels or contact links"),
        ("footprints", "footprint", [1, 0], "footprint pixels or contact links"),
        ("footprints", "side", ["R", "L"], "footprint pixels or contact links"),
    ],
)
def test_verifies_physical_rows(
    tmp_path, resources, verifier, name, column, values, message
):
    resource = resources[name]
    path = tmp_path / resource["zenodo_filename"]
    table = pq.read_table(path)
    table = replace_values(table, column, values)
    pq.write_table(table, path)
    resource.update(file_info(path))
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match=message):
        verifier(tmp_path)


def test_physical_row_count_includes_orphan_feature_row(tmp_path, resources, verifier):
    resource = resources["pressure_features"]
    path = tmp_path / resource["zenodo_filename"]
    table = pq.read_table(path)
    row = replace_values(table.slice(0, 1), "trial_id", [999])
    pq.write_table(pa.concat_tables([table, row]), path)
    resource.update(file_info(path))
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="Row count mismatch in pressure_features"):
        verifier(tmp_path)


def test_two_rows_cannot_describe_the_same_sensor_in_one_frame(
    tmp_path, resources, verifier
):
    path = tmp_path / "pressure.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table.slice(0, 1), table.slice(0, 1)]), path)
    resources["pressure"].update(file_info(path))
    path = tmp_path / "trials.parquet"
    trials = pq.read_table(path)
    pq.write_table(
        replace_values(trials, "n_cells", [2, 0]),
        path,
    )
    resources["trials"].update(file_info(path))
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="sensor"):
        verifier(tmp_path)


@pytest.mark.parametrize("split", [False, True])
def test_running_contact_is_one_uninterrupted_loaded_interval(verifier, split):
    from scripts.verify_release import verify_contacts

    # The same foot is either active in consecutive frames or separated by flight.
    second_frame = 1 if split else 2
    intervals = "(1, 0, 1, 1), (2, 1, 2, 1)" if split else "(1, 0, 3, 2)"
    with duckdb.connect() as con:
        con.execute(
            "CREATE TEMP TABLE trials AS SELECT 0 trial_id, 3 n_frames, 100 frequency_hz, 'rc' AS condition"
        )
        con.execute(
            """
            CREATE TEMP TABLE moments AS
            SELECT 0 trial_id, unnest([0, ?]) AS frame, 'L' support, 1 cells, 1 peak, 1 AS load, 0 ymin, 0 ymax
        """,
            [second_frame],
        )
        con.execute(
            f"""
            CREATE TEMP TABLE contacts AS
            SELECT 0 trial_id, contact, 'L' side, start_frame, stop_frame,
                   (stop_frame - start_frame) / 100 AS contact_time, 1 peak_pressure,
                   8.469^2 / 1000 AS peak_grf_z, n * 8.469^2 / 100000 AS vertical_impulse,
                   8.469^2 / 100 AS max_contact_area, 0 contact_y_min, 0 contact_y_max
            FROM (VALUES {intervals}) AS samples(contact, start_frame, stop_frame, n)
        """
        )
        con.execute("CREATE TEMP MACRO agrees(a, b) AS abs(a - b) <= 1e-8")
        with pytest.raises(ValueError, match="contact"):
            verify_contacts(con)


@pytest.mark.parametrize("field", ["path", "zenodo_filename"])
def test_two_logical_resources_require_two_files(tmp_path, resources, verifier, field):
    resources["footprints"][field] = resources["pressure_features"][field]
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="Duplicate"):
        verifier(tmp_path)


def test_cohort_accepts_changed_measurement_count_when_physical_tables_agree(
    tmp_path, resources, verifier
):
    # Add a second, adjacent sensor to the left-foot contact of the first trial.
    changes = {}
    pressure = pq.read_table(tmp_path / "pressure.parquet")
    cell = replace_values(pressure.slice(0, 1), "x", [1])
    changes["pressure"] = pa.concat_tables([pressure, cell])
    trials = pq.read_table(tmp_path / "trials.parquet")
    changes["trials"] = replace_values(trials, "n_cells", [2, 1])
    frames = pq.read_table(tmp_path / resources["pressure_features"]["zenodo_filename"])
    rows = frames.to_pylist()
    for side in ("", "_left"):
        rows[0]["grf_z" + side] *= 2
        rows[0]["contact_area" + side] *= 2
        rows[0]["cop_x" + side] = 8.469 / 2
    changes["pressure_features"] = pa.Table.from_pylist(rows, schema=frames.schema)
    contacts = pq.read_table(tmp_path / resources["contacts"]["zenodo_filename"])
    rows = contacts.to_pylist()
    for column in ("peak_grf_z", "max_contact_area", "vertical_impulse"):
        rows[0][column] *= 2
    changes["contacts"] = pa.Table.from_pylist(rows, schema=contacts.schema)
    for name, table in changes.items():
        resource = resources[name]
        path = tmp_path / resource["zenodo_filename"]
        pq.write_table(table, path)
        resource.update(file_info(path), rows=len(table))
    write_manifest(tmp_path, resources)
    assert verifier(tmp_path)["loaded_sensor_readings"] == 3


@pytest.mark.parametrize(
    "damage",
    [
        "extra_descriptor",
        "missing_descriptor",
        "wrong_pressure_type",
        "missing_dictionary",
        "dictionary_type",
    ],
)
def test_public_table_schema_is_independent_of_manifest(
    tmp_path, resources, verifier, damage
):
    name = "trials" if damage.endswith("descriptor") else "pressure"
    resource = resources[name]
    path = tmp_path / resource["zenodo_filename"]
    table = pq.read_table(path)
    if damage == "extra_descriptor":
        table = table.append_column("DateOfBirth", [["private", "private"]])
    elif damage == "missing_descriptor":
        table = table.drop_columns(["Age"])
    elif damage == "wrong_pressure_type":
        index = table.schema.get_field_index("value")
        table = table.set_column(index, "value", [[1.0, 1.0]])
    pq.write_table(table, path)
    resource.update(file_info(path))
    resource["columns"] = {f.name: {"type": str(f.type)} for f in table.schema}
    if damage == "missing_dictionary":
        del resource["columns"]
    elif damage == "dictionary_type":
        resource["columns"]["value"]["type"] = "double"
    write_manifest(tmp_path, resources)
    with pytest.raises(ValueError, match="schema|[Cc]olumn"):
        verifier(tmp_path)


@pytest.mark.parametrize("subjects", [SUBJECTS[:-1], [*SUBJECTS, "trial_id"]])
def test_verifier_checks_declared_subject_projection(
    tmp_path, resources, verifier, subjects
):
    write_manifest(tmp_path, resources, subject_columns=subjects)
    with pytest.raises(ValueError, match="subject_columns"):
        verifier(tmp_path)


def test_verifier_checks_side_coverage(tmp_path, resources, verifier):
    write_manifest(
        tmp_path,
        resources,
        pressure_features={
            "side_coverage": {"wc": {"frames": 3, "unresolved_per_foot_frames": 1}}
        },
    )
    with pytest.raises(ValueError, match="side_coverage"):
        verifier(tmp_path)


@pytest.mark.parametrize(
    ("field", "path"),
    [
        ("path", "../outside"),
        ("path", "metadata.json"),
        ("zenodo_filename", ".."),
        ("zenodo_filename", "metadata.json"),
    ],
)
def test_manifest_rejects_invalid_paths_before_reading_files(resources, field, path):
    from ppds.release_layout import validate_inventory

    resources["pressure"][field] = path
    with pytest.raises(ValueError):
        validate_inventory(resources)
