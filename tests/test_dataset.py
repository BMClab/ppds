"""Exercise the PPDS scientific accessors against the prepared release and an HTTP mirror."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from reader_data import Release

from ppds import dataset


def test_subjects_has_every_subject(source):
    subjects = dataset.subjects()
    assert subjects.subject.tolist() == [f"s{n:03d}" for n in range(1, 96)]
    assert not subjects.isna().any().any()
    assert "Nenhum" not in set(subjects["FootOrthosis"])
    assert "High-heels" not in set(subjects["Footwear"])
    for column in subjects.columns:
        if subjects[column].dtype == object or str(subjects[column].dtype).startswith("str"):
            assert (subjects[column] == subjects[column].str.strip()).all(), column


def test_subjects_query_selects_by_descriptors(source):
    older_women = dataset.subjects("AgeGroup == 'O' and Gender == 'F'")
    assert len(older_women) == 22
    assert set(older_women.AgeGroup) == {"O"} and set(older_women.Gender) == {"F"}


def test_trials_count_and_filters(source):
    all_trials = dataset.trials()
    assert len(all_trials) == all_trials.trial_id.nunique() == 3312
    assert all_trials.groupby("condition").size().to_dict() == {"ws": 950, "wc": 948, "wf": 949, "rc": 370, "ss": 95}
    assert dataset.trials(condition="rc").subject.nunique() == 37
    assert len(dataset.trials(subject="s041", condition="wf")) == 10
    assert set(dataset.trials(subject=["s001", "s002"], condition=["ss", "rc"]).condition) == {"ss", "rc"}
    assert dataset.trials(subject=[]).empty


def test_trials_selected_by_subject_descriptors(source):
    trials = dataset.trials(condition="wf", query="AgeGroup == 'O' and Gender == 'F'")
    assert len(trials) == 219


def test_pressure_returns_frames_y_x_in_kpa(source):
    p = dataset.pressure("s001", "wc", 1)
    assert p.shape == (213, 353, 64)
    assert p.dtype == np.uint16
    assert p[50, 49, 36] == 300
    assert p.max() == 355


def test_pressure_of_missing_trial_raises(source):
    with pytest.raises(LookupError):
        dataset.pressure("s022", "wc", 10)


def test_missing_pressure_feature_filters_return_the_same_typed_table(source):
    missing = dataset.pressure_features("s022", "wc", 10)
    assert missing.shape == (0, 20)
    assert missing.frame.dtype == np.uint16
    pd.testing.assert_frame_equal(missing, dataset.pressure_features("s022", "wc", [10]))
    pd.testing.assert_frame_equal(missing, dataset.pressure_features(["s022"], "wc", 10))


def test_pressure_rejects_unknown_side():
    with pytest.raises(ValueError, match="Expected side"):
        dataset.pressure("s001", "wc", 1, side="X")


def test_peak_image_is_max_over_frames(source):
    np.testing.assert_array_equal(dataset.peak_image("s001", "wc", 1), dataset.pressure("s001", "wc", 1).max(axis=0))


def test_standing_reads_full_platform_series(source):
    grid = dataset.standing("s001")
    assert grid.shape == (100, 353, 64) and grid.dtype == np.uint16
    # Independent samples and nonzero count from the selected raw quiet-stance series.
    assert grid[0, 29, 18] == 90
    assert grid[99, 30, 17] == 85
    assert np.count_nonzero(grid) == 25_643
    trial = dataset.trials("s001", "ss").iloc[0]
    assert trial.frequency_hz == 10 and trial.duration_s == 10 and trial.n_cells == 25_643


@pytest.mark.parametrize("condition", ["wc", "ss"])
def test_pressure_features_match_dense_frames(source, condition):
    feature = dataset.pressure_features("s001", condition, 1)
    grid = dataset.pressure("s001", condition, 1)
    assert feature.frame.tolist() == list(range(len(grid)))
    # kPa times sensor area in cm² times 0.1 gives force in N.
    np.testing.assert_allclose(feature.grf_z, grid.sum(axis=(1, 2)) * dataset.CELL_CM**2 * 0.1,
                               rtol=1e-5, atol=1e-5)
    assert feature.trial.eq(1).all() and feature.subject.eq("s001").all()


@pytest.mark.parametrize("condition", ["wc", "rc", "ss"])
def test_sides_reconstruct_measured_pressure_and_frame_features(source, condition):
    selection = ("s001", condition, 1)
    combined = dataset.pressure(*selection)
    left = dataset.pressure(*selection, side="l")
    right = dataset.pressure(*selection, side="R")
    assert left.dtype == right.dtype == np.uint16
    assert not np.any((left > 0) & (right > 0))
    np.testing.assert_array_equal(left + right, combined)
    frames = dataset.pressure_features(*selection)
    for grid, suffix in ((combined, ""), (left, "_left"), (right, "_right")):
        np.testing.assert_array_equal(frames["peak_pressure" + suffix], grid.max(axis=(1, 2)))
        np.testing.assert_allclose(frames["contact_area" + suffix], np.count_nonzero(grid, axis=(1, 2)) * dataset.CELL_CM**2)
        np.testing.assert_allclose(frames["grf_z" + suffix], grid.sum(axis=(1, 2)) * dataset.CELL_CM**2 * 0.1)
        for coord, axis in (("cop_x", 1), ("cop_y", 2)):
            weights = grid.sum(axis=axis)
            load = weights.sum(axis=1)
            moment = weights @ (np.arange(weights.shape[1]) * dataset.CELL_CM * 10)
            expected = np.divide(moment, load, out=np.full(len(load), np.nan), where=load > 0)
            np.testing.assert_allclose(frames[coord + suffix], expected, equal_nan=True)


def test_contacts_match_selected_trial_and_walking_footprints(source):
    contacts = dataset.contacts("s001", "wc", 1)
    assert contacts.contact.tolist() == [1, 2, 3, 4]
    assert contacts.side.tolist() == ["R", "L", "R", "L"]
    np.testing.assert_allclose(contacts.contact_time, (contacts.stop_frame - contacts.start_frame) / 100)
    for side, steps in contacts.groupby("side"):
        grid = dataset.pressure("s001", "wc", 1, side=side)
        for step in steps.itertuples(index=False):
            images = grid[step.start_frame:step.stop_frame]
            force = images.sum(axis=(1, 2)) * dataset.CELL_CM**2 * 0.1
            y = np.nonzero(images.max(axis=0))[0]
            assert step.peak_pressure == images.max()
            assert step.peak_grf_z == pytest.approx(force.max())
            assert step.vertical_impulse == pytest.approx(force.sum() / 100)
            assert step.max_contact_area == pytest.approx(np.count_nonzero(images, axis=(1, 2)).max() * dataset.CELL_CM**2)
            np.testing.assert_allclose([step.contact_y_min, step.contact_y_max], np.array([y.min(), y.max()]) * dataset.CELL_CM * 10)
    assert dataset.contacts("s001", "ss", 1).empty
    assert dataset.contacts(subject=[]).empty
    with dataset.connect() as con:
        rows = con.sql("""
            SELECT DISTINCT f.trial_id, f.contact, f.side, f.footprint
            FROM footprints f JOIN contacts c USING (trial_id, contact, side)
            JOIN trials t USING (trial_id)
            WHERE t.subject = 's001' AND t.condition = 'wc' AND t.trial = 1
            ORDER BY f.contact
        """).df()
    pd.testing.assert_frame_equal(rows[["trial_id", "contact", "side"]], contacts[["trial_id", "contact", "side"]])


def test_contact_image_reproduces_sensor_and_registered_peaks(source):
    key = ("s001", "wc", 1)
    steps = dataset.contacts(*key)
    # Contact 2 is on the left during a trial that also has right-foot load.
    left = steps.loc[steps.contact == 2].iloc[0]
    series = dataset.pressure(*key, side="L")
    measured = series[left.start_frame:left.stop_frame].max(axis=0)
    np.testing.assert_array_equal(dataset.contact_image(*key, 2), measured)
    # This is the first left contact of the first comfortable-walking trial.
    np.testing.assert_array_equal(
        dataset.contact_image(*key, 2, registered=True),
        dataset.footprints("s001", "wc", "L")[0],
    )
    run = dataset.contacts("s081", "rc", 4).iloc[0]
    assert dataset.contact_image("s081", "rc", 4, 1).max() == run.peak_pressure
    with pytest.raises(LookupError, match="walking"):
        dataset.contact_image("s081", "rc", 4, 1, registered=True)
    for args in [("s001", "ss", 1, 1), (*key, 20)]:
        with pytest.raises(LookupError, match="contact"):
            dataset.contact_image(*args)
    np.testing.assert_array_equal(dataset.contact_image(*key, "2"), measured)
    for bad in (1.5, True, "0", "abc"):
        with pytest.raises(ValueError, match="positive integer"):
            dataset.contact_image(*key, bad)


@pytest.mark.parametrize("lookup, args, kwargs", [
    (dataset.pressure, (["s001"], "wc", 1), {}),
    (dataset.peak_image, ("s001", 1, 1), {}),
    (dataset.contact_image, (["s001"], "wc", 1, 1), {}),
    (dataset.contact_image, ("s001", ["wc"], 1, 1), {"registered": True}),
])
def test_single_trial_requires_scalar_subject_and_condition(lookup, args, kwargs):
    with pytest.raises(TypeError, match="subject and condition"):
        lookup(*args, **kwargs)


def test_footprints_of_one_side(source):
    f = dataset.footprints("s001", "wc", "L")
    assert f.shape == (20, 353, 64)
    np.testing.assert_array_equal(f, dataset.footprints("s001", "wc", "l"), strict=True)
    assert dataset.footprints("s001", "ss", "L").shape == (0, 353, 64)


@pytest.mark.parametrize("selection", [
    (["s001", "s002"], "wc", "L"), ("s001", None, "L"),
    ("s001", ["wc", "wf"], "L"), ("s001", "wc", ["L", "R"]), ("s001", "wc", "X"),
])
def test_dense_footprints_require_one_group(selection):
    with pytest.raises((TypeError, ValueError), match="Footprints require|Expected side"):
        dataset.footprints(*selection)


@pytest.mark.parametrize("trial", [1.9, True, np.bool_(True), 0, -1])
def test_trial_numbers_cannot_select_a_different_trial(trial):
    with pytest.raises(ValueError, match="positive integer"):
        dataset.peak_image("s001", "wc", trial)


def test_invalid_trial_in_multi_value_filter(source):
    with pytest.raises(ValueError, match="positive integer"):
        dataset.contacts(subject="s001", trial=[1, 2.5])


@pytest.mark.parametrize("selector", [
    np.ma.array(1, mask=True),
    np.array(np.timedelta64(1, "ns")),
    np.array(np.datetime64(1, "ns")),
])
def test_masked_and_temporal_ids_cannot_select_contacts(source, selector):
    with pytest.raises(ValueError, match="Trial must be a positive integer"):
        dataset.contacts("s001", "wc", selector)
    with pytest.raises(ValueError, match="Contact must be a positive integer"):
        dataset.contact_image("s001", "wc", 1, selector)


@pytest.mark.parametrize("bad_subject", [True, np.bool_(True), 123, b"s001", ["s001", False], ["s001", 123]])
def test_invalid_filter_values_are_rejected(source, bad_subject):
    with pytest.raises(TypeError, match="Invalid filter value"):
        dataset.trials(subject=bad_subject)
    with pytest.raises(TypeError, match="Invalid filter value"):
        dataset.trials(subject=bad_subject, query="Age > 0")


def test_blank_query_and_zero_dim_trial_arrays(source):
    pd.testing.assert_frame_equal(dataset.subjects("   "), dataset.subjects())
    pd.testing.assert_frame_equal(dataset.trials(query="   "), dataset.trials())
    with pytest.raises(TypeError, match="Query"):
        dataset.subjects(123)
    with pytest.raises(TypeError, match="Query"):
        dataset.trials(query=123)
    pd.testing.assert_frame_equal(
        dataset.pressure_features("s001", "wc", np.array(1)),
        dataset.pressure_features("s001", "wc", 1),
    )
    with pytest.raises(ValueError, match="positive integer"):
        dataset.pressure_features("s001", "wc", np.array(True))


def test_private_vendor_tables_are_absent_from_the_public_interface(source):
    private_tables = {"spatiotemporal", "grf", "grf_cycle"}
    with dataset.connect() as con:
        assert private_tables.isdisjoint(row[0] for row in con.execute("SHOW TABLES").fetchall())
    assert private_tables.isdisjoint(dataset.TABLES)
    assert all(not hasattr(dataset, name) for name in private_tables | {"grf_trials"})


def test_scan_vertices_and_faces(source):
    vertices, faces = dataset.scan("s001", "L")
    assert faces.shape == (91361, 3)
    assert faces.max() < len(vertices)
    assert np.ptp(vertices, axis=0) == pytest.approx([0.242, 0.098, 0.155], abs=0.001)


def test_sql_views_match_release_counts(source):
    with dataset.connect() as con:
        assert {row[0] for row in con.execute("SHOW TABLES").fetchall()} == set(dataset.TABLES)
        assert con.execute("SELECT count(*) FROM subjects").fetchone()[0] == 95
        assert con.execute("SELECT count(*) FROM pressure").fetchone() == con.execute("SELECT sum(n_cells) FROM trials").fetchone()
        assert con.execute("SELECT count(*) FROM pressure_features").fetchone() == con.execute(
            "SELECT sum(n_frames) FROM trials").fetchone()
        assert con.execute("SELECT count(*) FROM standing").fetchone() == con.execute(
            "SELECT sum(n_cells) FROM trials WHERE condition='ss'").fetchone()
        assert con.execute("""SELECT count(*) FROM trials t JOIN subjects s USING (subject)
                          WHERE s.AgeGroup='O' AND s.Gender='F' AND t.condition='wf'""").fetchone() == (219,)


@pytest.mark.parametrize(("subject", "condition"), [("s016", "rc"), ("s041", "wf")])
def test_conditions_have_pressure_and_frame_features(source, subject, condition):
    assert len(dataset.trials(subject=subject, condition=condition)) == 10
    with dataset.connect() as con:
        for table in ("pressure", "pressure_features"):
            n = con.execute(f"SELECT count(*) FROM {table} WHERE subject = ? AND condition = ?",
                            [subject, condition]).fetchone()[0]
            assert n > 0, table
    if condition == "wf":
        assert len(dataset.footprints(subject, condition, "L")) == 20
        assert len(dataset.footprints(subject, condition, "R")) == 21


def test_zenodo_locations_are_flat_file_urls(tmp_path):
    try:
        dataset.use_zenodo(1234567)
        assert dataset.location("features/pressure_features.parquet") == (
            "https://zenodo.org/api/records/1234567/files/features_pressure_features.parquet/content")
        dataset.use_local(tmp_path)
        assert dataset.location("features/pressure_features.parquet") == str(tmp_path / "features/pressure_features.parquet")
    finally:
        dataset.use_local()


def test_installed_local_default_follows_working_directory(tmp_path):
    # Load the package outside src/, as in a wheel installation.
    packages = tmp_path / "site-packages"
    shutil.copytree(Path(dataset.__file__).parent, packages / "ppds", ignore=shutil.ignore_patterns("__pycache__"))
    study = tmp_path / "study"
    study.mkdir()
    code = """
import os
from pathlib import Path
from ppds import dataset

assert Path(dataset.location("metadata.json")) == Path.cwd() / "data" / "metadata.json"
os.environ.pop("PPDS_DATA")
os.chdir("study")
dataset.use_local()
assert Path(dataset.location("metadata.json")) == Path.cwd() / "data" / "metadata.json"
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": str(packages), "PPDS_DATA": ""})


def test_remote_reads_transfer_only_the_parts_needed(http_root, data_root):
    url, server = http_root
    local = Release.local(data_root)
    dataset.use_url(url, flat=True)
    try:
        for call, path in (
            (lambda: dataset.footprints("s001", "wc", "L"), "features/footprints.parquet"),
            (lambda: dataset.pressure("s045", "wc", 3), "pressure.parquet"),
            (lambda: dataset.pressure("s045", "wc", 3, side="R"), "pressure.parquet"),
            (lambda: dataset.pressure_features("s045", "wc", 3), "features/pressure_features.parquet"),
            (lambda: dataset.contact_image("s045", "wc", 3, 1), "pressure.parquet"),
            (lambda: dataset.contact_image("s045", "wc", 3, 1, registered=True), "features/footprints.parquet"),
            (lambda: dataset.scan("s053", "R"), "scans.zip"),
        ):
            name = path.replace("/", "_")
            before, ranges = server.sent[name], server.ranges[name]
            call()
            assert 0 < server.sent[name] - before < Path(local.location(path)).stat().st_size // 4, path
            assert server.ranges[name] > ranges, path
    finally:
        dataset.use_local()


def test_subject_projection_rejects_disagreement_and_preserves_nulls(tmp_path, monkeypatch):
    path = tmp_path / "index.parquet"
    metadata = {"subject_columns": ["subject", "age-year"],
                "resources": {"trials": {"path": path.name}}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    rows = pd.DataFrame({"subject": ["s001", "s001", "s002"], "age-year": [None, None, 40.]})
    rows.to_parquet(path)
    monkeypatch.setenv("PPDS_DATA", str(tmp_path))
    dataset.use_local()
    try:
        subjects = dataset.subjects()
        assert subjects.subject.tolist() == ["s001", "s002"] and pd.isna(subjects.loc[0, "age-year"])
        rows.loc[1, "age-year"] = 50.
        rows.to_parquet(path)
        with pytest.raises(ValueError, match="s001"):
            dataset.subjects()
    finally:
        monkeypatch.undo()
        dataset.use_local()


@pytest.mark.parametrize("has_contacts", [True, False])
def test_sql_and_reader_follow_release_schema(tmp_path, has_contacts):
    trials = pd.DataFrame({"trial_id": [0, 1], "subject": ["s001", "s002"], "condition": ["ws", "wf"],
                           "trial": [1, 1], "n_frames": [3, 3]})
    cells = pd.DataFrame({"trial_id": [0, 0, 1], "frame": [0, 0, 1], "y": [7, 7, 0], "x": [8, 5, 2],
                          "value": [60, 35, 8]})
    tables = {"trials": trials, "pressure": cells, "pressure_features": trials[["trial_id"]],
              "footprints": pd.DataFrame({"footprint": [0]})}
    if has_contacts:
        cells["side"] = ["L", "R", "R"]
        tables["contacts"] = pd.DataFrame({"trial_id": [1, 0, 0], "contact": [1, 2, 1], "side": ["R", "L", "R"]})
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
    resources = {name: {"path": f"{name}.parquet"} for name in tables}
    (tmp_path / "metadata.json").write_text(json.dumps({"subject_columns": ["subject"], "resources": resources}))
    dataset.use_local(tmp_path)
    try:
        with dataset.connect() as con:
            assert ("contacts" in {row[0] for row in con.execute("SHOW TABLES").fetchall()}) == has_contacts
            if has_contacts:
                sql = con.sql("SELECT c.* FROM contacts c JOIN trials t USING (trial_id) WHERE t.subject = 's001' ORDER BY contact").df()
                result = dataset.contacts(["s001"], "ws", 1)
                pd.testing.assert_frame_equal(sql, result)
                assert result.contact.tolist() == [1, 2]
                left = dataset.pressure("s001", "ws", 1, side="L")
                assert left.shape == (3, 353, 64) and np.count_nonzero(left) == 1 and left[0, 7, 8] == 60
        assert dataset.pressure("s001", "ws", 1).sum() == 95
    finally:
        dataset.use_local()


def test_small_release_selects_one_trial_and_one_mesh_over_http(tmp_path):
    """Exercise the public transport on fresh checkouts without the large scientific release."""
    import struct
    import zipfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from conftest import range_server

    resources = {name: {"path": path} for name, path in (("trials", "index.parquet"), ("pressure", "cells.parquet"),
                                                          ("scans", "objects.zip"))}
    rows = []
    y, x = np.indices((353, 64))
    n_cells = y.size
    with pq.ParquetWriter(tmp_path / "cells.parquet", pa.schema([
        ("trial_id", pa.uint16()), ("frame", pa.uint16()), ("y", pa.uint16()),
        ("x", pa.uint8()), ("value", pa.uint16()),
    ]), compression="NONE", use_dictionary=False) as writer:
        for i in range(32):
            subject = f"s{i + 1:03d}"
            rows.append({"subject": subject, "condition": "wc", "trial": 1, "trial_id": i, "n_frames": 2})
            writer.write_table(pa.table({
                "trial_id": np.full(n_cells, i, np.uint16), "frame": np.ones(n_cells, np.uint16),
                "y": y.ravel().astype(np.uint16), "x": x.ravel().astype(np.uint8),
                "value": np.full(n_cells, 10 + i, np.uint16),
            }))
    pd.DataFrame(rows).to_parquet(tmp_path / "index.parquet")
    triangle = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    with zipfile.ZipFile(tmp_path / "objects.zip", "w") as archive:
        for side, count in (("l", 1), ("r", 50_000)):
            archive.writestr(f"scans/s010{side}_foot.stl", bytes(80) + struct.pack("<I", count) + triangle * count)
    resources["scans"]["bytes"] = (tmp_path / "objects.zip").stat().st_size
    (tmp_path / "metadata.json").write_text(json.dumps({"subject_columns": ["subject"], "resources": resources}))
    with range_server({p.name: p for p in tmp_path.iterdir()}) as (url, server):
        server.chunked_head.add("objects.zip")
        try:
            for local in (True, False):
                if local:
                    dataset.use_local(tmp_path)
                else:
                    dataset.use_url(url)
                grid = dataset.pressure("s010", "wc", 1)
                assert grid.shape == (2, 353, 64)
                assert np.all(grid[0] == 0) and np.all(grid[1] == 19)
                with pytest.raises(ValueError, match="per-sensor side labels"):
                    dataset.pressure("s010", "wc", 1, side="L")
                with pytest.raises(LookupError, match="gait contacts table"):
                    dataset.contacts("s010", "wc", 1)
                with pytest.raises(LookupError, match="pressure features table"):
                    dataset.pressure_features("s010", "wc", 1)
                with pytest.raises(LookupError, match="registered footprints table"):
                    dataset.footprints("s010", "wc", "L")
                vertices, faces = dataset.scan("s010", "l")
                assert vertices.shape == (3, 3) and faces.shape == (1, 3)
            # Also verify a pressure-only release reports LookupError for missing scans.
            no_scans = {k: v for k, v in resources.items() if k != "scans"}
            (tmp_path / "metadata.json").write_text(json.dumps({"subject_columns": ["subject"], "resources": no_scans}))
            dataset.use_local(tmp_path)
            with pytest.raises(LookupError, match="foot scans archive"):
                dataset.scan("s010", "l")
            for file in ("objects.zip", "cells.parquet"):
                assert server.ranges[file] > 0
                assert 0 < server.sent[file] < (tmp_path / file).stat().st_size // 4
        finally:
            dataset.use_local()


def test_notebook_source_prefers_explicit_record_then_local_release(tmp_path, monkeypatch):
    monkeypatch.setenv("PPDS_DATA", str(tmp_path))
    monkeypatch.setattr(dataset, "_source", dataset._source)
    dataset.use_notebook()
    assert dataset.location("trials.parquet") == str(tmp_path / "trials.parquet")
    dataset.use_notebook(12345)
    assert dataset.location("trials.parquet").endswith("12345/files/trials.parquet/content")
    monkeypatch.delenv("PPDS_DATA")
    monkeypatch.setattr(dataset, "local_data_dir", lambda: tmp_path)
    dataset.use_notebook()
    assert dataset.location("trials.parquet").endswith("24432612/files/trials.parquet/content")
    (tmp_path / "metadata.json").write_text("{}")
    dataset.use_notebook()
    assert dataset.location("trials.parquet") == str(tmp_path / "trials.parquet")
