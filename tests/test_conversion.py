"""Exercise the public pressure converters through saved files and PPDS queries."""

import json
from importlib.resources import files

import numpy as np
import pyarrow.parquet as pq
import pytest

from ppds import conversion, dataset


def test_both_npz_layouts_round_trip_with_trial_clocks(tmp_path):
    legacy = tmp_path / "s123wc012.npz"
    modern = tmp_path / "s456ss001.npz"
    data = np.zeros((353, 64, 5))
    data[78, 9, 2] = 7.3
    np.savez(legacy, data=data)
    np.savez(
        modern,
        data=np.zeros((7, 353, 64), np.float32),
        array_axes=["frame", "y", "x"],
        sampling_rate_hz=12.5,
    )
    out = conversion.npz_to_parquet(
        [modern, legacy], tmp_path / "tables", sampling_rate_hz=100
    )
    rows = pq.read_table(out / "trials.parquet").to_pylist()
    assert [(r["subject"], r["n_frames"], r["frequency_hz"]) for r in rows] == [
        ("s123", 5, 100),
        ("s456", 7, 12.5),
    ]
    assert pq.read_table(out / "pressure.parquet").to_pylist() == [
        {"trial_id": 0, "frame": 2, "y": 78, "x": 9, "value": 73}
    ]
    dataset.use_local(out)
    try:
        with dataset.connect() as con:
            assert con.sql("select subject, trial, value from pressure").fetchall() == [
                ("s123", 12, 73)
            ]
        assert dataset.pressure("s456", "ss", 1).shape == (7, 353, 64)
    finally:
        dataset.use_local()
    exported = conversion.parquet_to_npz(
        out, tmp_path / "export", trials=[("s123", "wc", 12)]
    )
    assert [p.name for p in exported] == [legacy.name]
    with np.load(exported[0]) as result:
        assert result.files == ["data"]
        np.testing.assert_array_equal(result["data"], data)
    with pytest.raises(FileExistsError):
        conversion.parquet_to_npz(out, exported[0].parent)
    with pytest.raises(LookupError):
        conversion.parquet_to_npz(out, tmp_path / "missing", trials=[("s123", "wc", 1)])
    assert not (tmp_path / "missing").exists()


def test_explicit_keys_and_validation_preserve_inputs(tmp_path):
    path = tmp_path / "recording.npz"
    key = ("walker", "fast", 301)
    custom = {key: path}
    data = np.zeros((2, 353, 64), np.float32)
    data[1, 4, 8] = 0.1
    np.savez(path, data=data)
    out = tmp_path / "tables"
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        conversion.npz_to_parquet(custom, out)
    assert not out.exists()
    conversion.npz_to_parquet(custom, out, sampling_rate_hz=300)
    assert pq.read_table(out / "pressure.parquet")["value"].to_pylist() == [1]
    data[1, 4, 8] = 0.15
    np.savez(path, data=data)
    with pytest.raises(ValueError, match="integer kPa"):
        conversion.npz_to_parquet(custom, tmp_path / "bad", sampling_rate_hz=100)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("name", ["press.txt", "pass001.npz", "s001_wc001.npz"])
def test_ambiguous_filenames_require_explicit_keys(tmp_path, name):
    path = tmp_path / name
    if path.suffix == ".txt":
        path.write_bytes(files("ppds").joinpath("samples/demowc.txt").read_bytes())
        convert = conversion.ascii_to_parquet
    else:
        np.savez(path, data=np.zeros((1, 353, 64)), sampling_rate_hz=100)
        convert = conversion.npz_to_parquet
    with pytest.raises(ValueError, match="supply a mapping"):
        convert(path, tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def test_ascii_example_selects_trials_and_survives_legacy_npz_round_trip(tmp_path):
    example = files("ppds").joinpath("samples/demowc.txt")
    out = conversion.ascii_to_parquet({("s000", "wc"): example}, tmp_path / "ascii")
    trial_rows = pq.read_table(out / "trials.parquet").to_pylist()
    assert [(r["subject"], r["trial"], r["frequency_hz"]) for r in trial_rows] == [
        ("s000", 1, 100),
        ("s000", 2, 100),
    ]
    cells = pq.read_table(out / "pressure.parquet")
    assert cells.num_rows > 0
    exported = conversion.parquet_to_npz(out, tmp_path / "original_layout")
    repeated = conversion.npz_to_parquet(
        exported, tmp_path / "from_npz", sampling_rate_hz=100
    )
    for name in ("trials", "pressure"):
        assert pq.read_table(out / f"{name}.parquet").equals(
            pq.read_table(repeated / f"{name}.parquet")
        )
    assert "resources" in json.loads((out / "metadata.json").read_text())


def test_saved_zebris_trials_convert_with_the_recorded_sampling_rate(tmp_path):
    from ppds.zebris import read_zebris_text

    raw = tmp_path / "s001wc.txt"
    raw.write_bytes(files("ppds").joinpath("samples/demowc.txt").read_bytes())
    read_zebris_text(raw)
    converted = conversion.npz_to_parquet(tmp_path, tmp_path / "from-npz")
    direct = conversion.ascii_to_parquet(raw, tmp_path / "from-ascii")
    for name in ("trials.parquet", "pressure.parquet"):
        assert pq.read_table(converted / name).equals(pq.read_table(direct / name))


def test_saving_zebris_trials_creates_outdir_and_never_overwrites_silently(tmp_path):
    from ppds.zebris import read_zebris_text

    raw = tmp_path / "s001wc.txt"
    raw.write_bytes(files("ppds").joinpath("samples/demowc.txt").read_bytes())
    outdir = tmp_path / "new" / "npz"
    _, trials, _ = read_zebris_text(raw, outdir=outdir)
    saved = sorted(outdir.glob("*.npz"))
    assert [p.name for p in saved] == ["s001wc001.npz", "s001wc002.npz"] and len(trials) == 2
    saved[1].write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="s001wc002.npz"):
        read_zebris_text(raw, outdir=outdir)
    assert saved[1].read_bytes() == b"keep"
    read_zebris_text(raw, outdir=outdir, overwrite=True)
    with np.load(saved[1]) as archive:
        np.testing.assert_array_equal(archive["data"], trials[1])


def test_ascii_timing_rotation_and_cleanup_are_applied_once(tmp_path):
    sample = files("ppds").joinpath("samples/demowc.txt").read_text()
    # Read the same two exercises with Windows line endings and a nonstandard filename.
    raw = tmp_path / "walking.txt"
    raw.write_bytes(sample.replace("\n", "\r\n").encode("ascii"))
    out = conversion.ascii_to_parquet({("p01", "walk"): raw}, tmp_path / "tables")
    meta = pq.read_table(out / "trials.parquet").to_pylist()
    assert [(r["n_frames"], r["duration_s"]) for r in meta] == [(6, 0.06), (6, 0.06)]
    exported = conversion.parquet_to_npz(out, tmp_path / "npz")
    for path, y, x in zip(exported, [35, 150], [18, 43], strict=True):
        with np.load(path) as trial:
            grid = trial["data"]
            assert grid.shape == (353, 64, 6)
            assert not grid[:, :, [0, 5]].any()
            assert grid[y, x, 1] == 4
            assert grid[y + 9, x, 4] == 7
    assert not list(tmp_path.glob("walking*.npz"))


def test_failed_batch_keeps_originals_and_publishes_no_partial_tables(tmp_path):
    good, bad = tmp_path / "s099wc001.npz", tmp_path / "s099wc002.npz"
    grid = np.zeros((353, 64, 1))
    np.savez(good, data=grid)
    np.savez(bad, data=grid, array_axes=["frame", "y", "x"])
    with pytest.raises(ValueError, match="array_axes"):
        conversion.npz_to_parquet(
            [good, bad], tmp_path / "tables", sampling_rate_hz=100
        )
    assert set(tmp_path.iterdir()) == {good, bad}


def test_non_pressure_ascii_reports_a_format_error(tmp_path):
    report = tmp_path / "s001wc.txt"
    report.write_text("[Ground reaction force]\nLeft force = {\n")
    with pytest.raises(ValueError, match="pressure export"):
        conversion.ascii_to_parquet(report, tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def test_ascii_manifest_tracks_recording_clock_through_trimming(tmp_path):
    sample = files("ppds").joinpath("samples/demowc.txt").read_text()
    # Include the unused acquisition frame at the beginning of exercise 1.
    sample = sample.replace("Walk A\t0.01\t0.06", "Walk A\t0.00\t0.07")
    path = tmp_path / "s001wc.txt"
    path.write_bytes(sample.replace("N/cm^2", "N/cm²").encode("latin-1"))
    output = conversion.ascii_to_parquet(path, tmp_path / "tables")
    clock = json.loads((output / "metadata.json").read_text())["ascii_exercises"]
    assert clock[0] == {
        "trial_id": 0,
        "start_frame": 0,
        "n_frames": 7,
        "first_retained_frame": 1,
        "recording_frames": 14,
    }
    assert pq.read_table(output / "trials.parquet")["n_frames"].to_pylist() == [6, 6]


@pytest.mark.parametrize(("first", "second", "expected"), [
    ("0.006\t0.066", "0.072\t0.068", [(1, 6), (7, 7)]),
    ("0.08\t0.06", "0.01\t0.06", [(8, 6), (1, 6)]),
    ("0.00\t0.09", "0.08\t0.06", [(0, 9), (8, 6)]),
])
def test_exercises_read_the_declared_frame_windows(tmp_path, first, second, expected):
    from ppds.zebris import open_zebris_trials

    sample = files("ppds").joinpath("samples/demowc.txt").read_text()
    path = tmp_path / "s001wc.txt"
    path.write_text(sample.replace("Walk A\t0.01\t0.06", "Walk A\t" + first)
                    .replace("Walk B\t0.08\t0.06", "Walk B\t" + second))
    with open_zebris_trials(path) as (_, trials):
        records = list(trials)
    assert [(t["start_frame"], t["n_frames"]) for _, _, t in records] == expected
    for (grid, _, _), (start, _) in zip(records, expected, strict=True):
        y, x = (150, 43) if start >= 7 else (35, 18)
        assert grid[1, y, x] == 4


def test_ascii_rejects_nonfinite_exercise_times(tmp_path):
    sample = files("ppds").joinpath("samples/demowc.txt").read_text()
    path = tmp_path / "s001wc.txt"
    path.write_text(sample.replace("Walk A\t0.01", "Walk A\tinf"))
    with pytest.raises(ValueError, match="exercise"):
        conversion.ascii_to_parquet(path, tmp_path / "tables")


def test_export_reads_standalone_parquet_pair(tmp_path):
    archive = tmp_path / "s001wc001.npz"
    grid = np.zeros((353, 64, 2))
    grid[26, 40, 0] = 3.1
    np.savez(archive, data=grid)
    tables = conversion.npz_to_parquet(archive, tmp_path / "pair", sampling_rate_hz=100)
    (tables / "metadata.json").unlink()
    (result,) = conversion.parquet_to_npz(tables, tmp_path / "export")
    with np.load(result) as restored:
        np.testing.assert_array_equal(restored["data"], grid)


def test_four_digit_trial_numbers_round_trip_by_filename(tmp_path):
    archive = tmp_path / "recording.npz"
    grid = np.zeros((2, 353, 64), dtype=np.float64)
    grid[1, 10, 12] = 4.5
    np.savez(archive, data=grid, sampling_rate_hz=100)
    tables = conversion.npz_to_parquet({("s001", "wc", 1000): archive}, tmp_path / "tables")
    (exported,) = conversion.parquet_to_npz(tables, tmp_path / "exported")
    assert exported.name == "s001wc1000.npz"
    reimported = conversion.npz_to_parquet(exported, tmp_path / "reimported", sampling_rate_hz=100)
    for name in ("trials.parquet", "pressure.parquet"):
        assert pq.read_table(tables / name).equals(pq.read_table(reimported / name))
