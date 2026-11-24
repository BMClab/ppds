"""Convert Zebris gait ASCII, PPDS trial NPZs and sparse pressure Parquet tables."""

import json
import re
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from reader_data import Release

from . import _pressure_tables as tables
from .dataset import CELL_CM
from .zebris import open_zebris_trials

PRESSURE_AXES = ("frame", "y", "x")


@contextmanager
def _new_directory(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "output"
        stage.mkdir()
        yield stage
        stage.rename(destination)


def _key(key, *, trial):
    length = 3 if trial else 2
    if not isinstance(key, tuple) or len(key) != length:
        raise ValueError(
            "Expected (subject, condition, trial) for NPZ or (subject, condition) for ASCII"
        )
    for part in key[:2]:
        if not isinstance(part, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", part
        ):
            raise ValueError(f"Invalid subject or condition in {key!r}")
    if trial and (
        not isinstance(key[2], Integral)
        or isinstance(key[2], bool)
        or not 1 <= key[2] <= 2**32 - 1
    ):
        raise ValueError(f"Invalid trial number in {key!r}")
    return key


def _inputs(files, *, trial):
    if isinstance(files, Mapping):
        pairs = [(_key(key, trial=trial), Path(path)) for key, path in files.items()]
    else:
        suffix = ".npz" if trial else ".txt"
        if isinstance(files, (str, Path)):
            path = Path(files)
            files = sorted(path.glob("*" + suffix)) if path.is_dir() else [path]
        pairs = []
        pattern = r"(s[0-9]{3})(wc|ws|wf|rc|ss)" + (r"([0-9]{3,})\.npz" if trial else r"\.txt")
        for path in map(Path, files):
            match = re.fullmatch(pattern, path.name)
            if match is None:
                raise ValueError(
                    f"Cannot infer the trial identity from {path.name}; supply a mapping of keys to paths"
                )
            subject, condition = match.groups()[:2]
            key = (subject, condition, int(match[3])) if trial else (subject, condition)
            pairs.append((_key(key, trial=trial), path))
    if not pairs or len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(
            "Provide at least one recording and a unique key for each input"
        )
    return sorted(pairs)


def _check_units(units):
    text = str(units).strip().lower().replace(" ", "").replace("²", "2")
    if text.replace("^", "") != "n/cm2":
        raise ValueError(f"Expected pressure in N/cm², received {units!r}")


def _npz_records(inputs, sampling_rate_hz):
    for key, path in inputs:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(
                    f"{path.name}: expected a prepared trial NPZ with a data array"
                )
            grid = archive["data"]
            if "pressure_units" in archive:
                _check_units(archive["pressure_units"].item())
            if grid.ndim != 3:
                raise ValueError(
                    f"{path.name}: expected a three-dimensional pressure grid"
                )
            if grid.shape[:2] == tables.GRID:
                axes = ("y", "x", "frame")
                grid = np.moveaxis(grid, -1, 0)
            elif grid.shape[1:] == tables.GRID:
                axes = PRESSURE_AXES
            else:
                raise ValueError(f"{path.name}: expected a 353 by 64 pressure grid")
            if "array_axes" in archive and tuple(archive["array_axes"]) != axes:
                raise ValueError(
                    f"{path.name}: array_axes disagrees with the grid shape"
                )
            if "sampling_rate_hz" in archive:
                hz = archive["sampling_rate_hz"].item()
            elif isinstance(sampling_rate_hz, Mapping):
                hz = sampling_rate_hz.get(key)
            else:
                hz = sampling_rate_hz
            if (
                not isinstance(hz, Real)
                or isinstance(hz, bool)
                or not np.isfinite(hz)
                or hz <= 0
            ):
                raise ValueError(
                    f"{path.name}: supply a positive sampling_rate_hz for {key!r}"
                )
            yield key, grid, hz, None


def _ascii_records(inputs):
    for (subject, condition), path in inputs:
        with open_zebris_trials(path, subject) as (info, trials):
            platform = info["platform"]
            _check_units(platform["units"])
            if (platform["ncols"], platform["nrows"]) != tables.GRID:
                raise ValueError(f"{path.name}: expected a 353 by 64 walkway")
            # Allow for sensor spacing rounded to 0.1 mm in a text header.
            pitch = (platform["cell_width"], platform["cell_height"])
            if not np.allclose(pitch, CELL_CM, atol=0.005, rtol=0):
                raise ValueError(
                    f"{path.name}: expected the PPDS sensor spacing of 8.469 mm"
                )
            for number, (grid, warnings, timing) in enumerate(trials, 1):
                timing["recording_frames"] = platform["nframes"]
                if warnings:
                    timing["boundary_warnings"] = [message for _, message in warnings]
                yield (subject, condition, number), grid, info["freq"], timing


def _write_tables(records, destination):
    exercises = []

    def encode():
        for trial_id, (key, grid, hz, timing) in enumerate(records):
            if trial_id > 65535:
                raise ValueError("Too many trials for the PPDS uint16 trial_id")
            if timing is not None:
                exercises.append({"trial_id": trial_id, **timing})
            yield tables.encode_pressure_trial(trial_id, *key, grid, hz)

    with _new_directory(destination) as stage:
        rows = tables._write_pressure(encode(), stage)
        indices = pa.Table.from_pylist(rows, schema=tables.TRIAL)
        pq.write_table(indices, stage / "trials.parquet", **tables.PARQUET)
        resources = {
            name: {"path": name + ".parquet"} for name in ("trials", "pressure")
        }
        manifest = {"resources": resources, "subject_columns": ["subject"]}
        if exercises:
            manifest["ascii_exercises"] = exercises
        (stage / "metadata.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return Path(destination)


def ascii_to_parquet(files, destination):
    """Convert gait ASCII exports to a new pressure collection and return its directory.

    Pass a path, directory, list of paths, or {(subject, condition): path}. With
    ordinary PPDS filenames (s001wc.txt) the identity comes from the filename.
    Each exercise uses its exported clock and the original notebook's trial
    cleanup; boundary warnings are recorded in metadata.json.
    """
    return _write_tables(_ascii_records(_inputs(files, trial=False)), destination)


def npz_to_parquet(files, destination, *, sampling_rate_hz=None):
    """Convert prepared N/cm² trial grids to a new collection and return its directory.

    Pass a path, directory, list of paths, or {(subject, condition, trial): path}.
    Both (353, 64, frames) and (frames, 353, 64) are accepted. An embedded
    sampling_rate_hz takes precedence over the supplied rate (a number or a
    mapping from trial keys). Input values must be representable as uint16 kPa.
    This converts pressure only; no trimming, foot assignment or rotation is applied.
    """
    return _write_tables(
        _npz_records(_inputs(files, trial=True), sampling_rate_hz), destination
    )


@contextmanager
def _pressure_tables(source):
    names = ("trials", "pressure")
    if isinstance(source, Release) or (Path(source) / "metadata.json").is_file():
        release = source if isinstance(source, Release) else Release.local(source)
        with release.connect(names) as con:
            yield con
    else:
        with duckdb.connect() as con:
            for name in names:
                con.read_parquet(
                    str(Path(source) / f"{name}.parquet"), hive_partitioning=False
                ).create_view(name)
            yield con


def parquet_to_npz(source, destination, *, trials=None):
    """Export selected (subject, condition, trial) keys, or all trials, to a new directory.

    source is a directory with trials.parquet and pressure.parquet, a local release
    directory, or a reader_data.Release. Return the written paths.
    Each NPZ has the original notebook's sole data array in N/cm²,
    ordered (y, x, frame). Sampling rates remain in the source trial index.
    """
    paths = []
    with (
        _pressure_tables(source) as con,
        _new_directory(destination) as stage,
    ):
        rows = con.execute(
            "SELECT subject, condition, trial, trial_id, n_frames, n_cells "
            "FROM trials ORDER BY trial_id"
        ).fetchall()
        index = {tuple(row[:3]): row[3:] for row in rows}
        if len(index) != len(rows):
            raise ValueError("Duplicate trial keys in the source index")
        selected = list(index) if trials is None else list(trials)
        if not selected:
            raise ValueError("Select at least one trial")
        for key in selected:
            subject, condition, number = _key(key, trial=True)
            if key not in index:
                raise LookupError(f"Unknown pressure trial {key!r}")
            name = f"{subject}{condition}{number:03d}.npz"
            if (stage / name).exists():
                raise ValueError(f"Duplicate output filename: {name}")
            trial_id, n_frames, n_cells = index[key]
            cells = con.execute(
                "SELECT frame, y, x, value FROM pressure WHERE trial_id = ?", [trial_id]
            ).fetchnumpy()
            frame, y, x, values = (cells[col] for col in ("frame", "y", "x", "value"))
            shape = (*tables.GRID, n_frames)
            if len(values) != n_cells:
                raise ValueError(f"Inconsistent sensor count for {key!r}")
            linear = np.ravel_multi_index((y, x, frame), shape)
            if np.unique(linear).size != n_cells:
                raise ValueError(f"Duplicate sensor coordinates for {key!r}")
            data = np.zeros(shape, dtype=np.float64)
            data.flat[linear] = values / 10
            np.savez_compressed(stage / name, data=data)
            paths.append(Path(destination) / name)
    return paths
