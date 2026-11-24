"""Read PPDS locally or over HTTP with DuckDB.

Use `use_local(directory)` for a downloaded release, `use_url(base)` for an
HTTP mirror, or `use_zenodo(record)` for a published version. By default the
reader uses PPDS_DATA when set. Otherwise it uses data/ in the source
checkout or, for a wheel installation, in the current working directory.

Pressure arrays are indexed [frame, y, x] and measured in integer kPa. Gait
is sampled at 100 Hz and quiet stance at 10 Hz. Trial lengths and frequencies
are available in trials(). SQL sessions opened by connect() support the
same logical tables as these helpers.
"""

import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import reader_data

from .paths import local_data_dir

NROWS, NCOLS = 353, 64
FREQ_HZ = 100.0
CELL_CM = 0.8469
TABLES = ("subjects", "trials", "standing", "footprints", "contacts",
          "pressure", "pressure_features")

_source = reader_data.Release.local(local_data_dir())


def use_local(path=None):
    """Use a release directory, defaulting to PPDS_DATA or the local data/ directory."""
    global _source
    _source = reader_data.Release.local(path if path is not None else local_data_dir())


def use_url(base, suffix="", flat=False):
    """Use an HTTP(S) release directory, optionally with flat names and a URL suffix."""
    global _source
    _source = reader_data.Release.http(base, suffix=suffix, flat=flat)


def use_zenodo(record):
    """Use the public release in a numbered Zenodo record."""
    global _source
    _source = reader_data.Release.zenodo(record)


def use_notebook(record=None, *, pressure_sample=False):
    """Select an explicit record, a local release, or the published PPDS showcase."""
    if record is not None:
        use_zenodo(record)
    elif os.environ.get("PPDS_DATA") or (local_data_dir() / "metadata.json").is_file():
        use_local()
    elif sys.platform == "emscripten" and pressure_sample:
        import marimo as mo

        from ._browser_sample import load

        use_local(load().name)
        mo.output.append(mo.md(
            "**Browser sample:** all 164 trials for four participants "
            "(s001, s005, s055, s058). Open in molab or run locally for the full 95-participant release."
        ))
    else:
        use_zenodo(24432612)


def location(rel):
    """Resolve a release-relative filename such as 'features/pressure_features.parquet'."""
    return _source.location(rel)


def _subject_view(con, columns):
    if "subject" not in columns or len(set(columns)) != len(columns):
        raise ValueError("subject_columns must include subject and contain unique column names")
    identifiers = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    con.execute(f"CREATE VIEW subjects AS SELECT DISTINCT {identifiers} FROM trials")
    invalid = con.execute(
        "SELECT subject FROM subjects GROUP BY subject HAVING subject IS NULL OR count(*) > 1 LIMIT 1"
    ).fetchone()
    if invalid:
        raise ValueError(f"Missing subject ID or inconsistent participant descriptors for {invalid[0]!r}")


def _participant_rows(con, query):
    _subject_view(con, _source.metadata["subject_columns"])
    frame = con.sql("SELECT * FROM subjects ORDER BY subject").df()
    if query is not None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        query = query.strip()
    return frame.query(query).reset_index(drop=True) if query else frame


def subjects(query=None):
    """One validated descriptor row per participant. query is a pandas expression."""
    with _source.connect("trials") as con:
        return _participant_rows(con, query)


def _with_trial_keys(con, name):
    keys = con.table("trials").project("trial_id, subject, condition, trial")
    _source.table(name, connection=con).join(keys, "trial_id").create_view(name)


def _positive_int(value, label):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # Scalar indexing preserves a missing value's mask and a temporal value's dtype.
        value = value[()]
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if isinstance(value, (bool, np.timedelta64)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _trial_number(value):
    return _positive_int(value, "Trial")


def _string_filter(name, value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, bool, np.bool_, int, float, np.number)):
        raise TypeError(f"Invalid filter value for {name}: {value!r}")
    items = list(value)
    if any(not isinstance(item, str) for item in items):
        raise TypeError(f"Invalid filter value for {name}: {value!r}")
    return items


def _require_resource(name, label):
    if name not in _source.metadata["resources"]:
        raise LookupError(f"This release does not contain {label}")


def _side_code(side):
    if not isinstance(side, str) or side.upper() not in ("L", "R"):
        raise ValueError("Expected side 'L' or 'R'")
    return side.upper()


def _select_rows(con, name, *, order=None, **filters):
    """Apply scalar and multi-value filters in DuckDB before collecting rows."""
    clauses, values = [], []
    for column, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Invalid filter value for {column}: {value!r}")
        if isinstance(value, (str, int, np.integer)):
            clauses.append(f"{column} = ?")
            values.append(int(value) if isinstance(value, np.integer) else value)
        else:
            items = list(value)
            if any(isinstance(item, (bool, np.bool_)) for item in items):
                raise TypeError(f"Invalid filter value for {column}: {value!r}")
            clauses.append(f"{column} IN (SELECT unnest(?))")
            values.append(items)
    sql = f"SELECT * FROM {name}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order:
        sql += " ORDER BY " + order
    return con.execute(sql, values).df()


def trials(subject=None, condition=None, query=None):
    """One row per pressure trial, including quiet stance (ss) and participant descriptors."""
    if query is not None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        query = query.strip()
    if subject is not None:
        subject = _string_filter("subject", subject)
    if condition is not None:
        condition = _string_filter("condition", condition)
    with _source.connect("trials") as con:
        if query:
            selected = _participant_rows(con, query).subject
            if subject is not None:
                selected = selected[selected.isin([subject] if isinstance(subject, str) else subject)]
            subject = selected.tolist()
        return _select_rows(con, "trials", subject=subject, condition=condition, order="trial_id")


def _find_trial(con, subject, condition, trial):
    return con.execute(
        "SELECT trial_id, n_frames FROM trials WHERE subject = ? AND condition = ? AND trial = ?",
        [subject, condition, trial],
    ).fetchone()


def _trial(subject, condition, trial):
    if not isinstance(subject, str) or not isinstance(condition, str):
        raise TypeError("A single trial requires subject and condition to be strings")
    number = _trial_number(trial)
    with _source.connect("trials") as con:
        row = _find_trial(con, subject, condition, number)
    if row is None:
        raise LookupError(f"no pressure data for {subject} {condition} trial {number}")
    return row


def pressure(subject, condition, trial, *, side=None):
    """Return uint16 [frame, y, x] in kPa, optionally for just the L or R foot.

    All selections retain the full trial clock and platform grid. Unloaded cells
    are zero. Per-foot selection requires a release with side-labelled cells.
    """
    if side is not None:
        side = _side_code(side)
    trial_id, n_frames = _trial(subject, condition, trial)
    with _source.connect("pressure") as con:
        sql = "SELECT frame, y, x, value FROM pressure WHERE trial_id = ?"
        params = [trial_id]
        if side is not None:
            if "side" not in con.table("pressure").columns:
                raise ValueError("This release does not contain per-sensor side labels")
            sql += " AND side = ?"
            params.append(side)
        cells = con.execute(sql, params).fetchnumpy()
    grid = np.zeros((n_frames, NROWS, NCOLS), np.uint16)
    grid[cells["frame"], cells["y"], cells["x"]] = cells["value"]
    return grid


def peak_image(subject, condition, trial):
    """Return each sensor's maximum across the trial as a uint16 [y, x] image in kPa."""
    trial_id, _ = _trial(subject, condition, trial)
    with _source.connect("pressure") as con:
        cells = con.execute(
            "SELECT y, x, max(value) AS peak FROM pressure WHERE trial_id = ? GROUP BY y, x", [trial_id]
        ).fetchnumpy()
    grid = np.zeros((NROWS, NCOLS), np.uint16)
    grid[cells["y"], cells["x"]] = cells["peak"]
    return grid


def standing(subject):
    """The participant's 100-frame quiet-stance trial at 10 Hz, uint16 [frame, y, x] in kPa.

    The recording contains both feet in platform coordinates. standing(subject)[t]
    returns the pressure map at t / 10 seconds from the start of the selected segment.
    """
    return pressure(subject, "ss", 1)


def _trial_measurements(name, subject, condition, trial, order):
    if subject is not None:
        subject = _string_filter("subject", subject)
    if condition is not None:
        condition = _string_filter("condition", condition)
    if trial is not None:
        if np.isscalar(trial) or (isinstance(trial, np.ndarray) and trial.ndim == 0):
            trial = _trial_number(trial)
        else:
            trial = [_trial_number(v) for v in trial]
    with _source.connect("trials") as con:
        _with_trial_keys(con, name)
        if (name == "pressure_features" and isinstance(subject, str)
                and isinstance(condition, str) and isinstance(trial, int)):
            # A physical trial_id predicate prunes Parquet row groups on remote reads.
            row = _find_trial(con, subject, condition, trial)
            if row is None:
                return con.table(name).limit(0).df()
            trial_id, _ = row
            return _select_rows(con, name, trial_id=trial_id, order=order)
        return _select_rows(con, name, subject=subject, condition=condition, trial=trial, order=order)


def pressure_features(subject=None, condition=None, trial=None):
    """Per-frame force, COP, contact area and peak pressure for both feet and each side."""
    _require_resource("pressure_features", "a pressure features table")
    return _trial_measurements("pressure_features", subject, condition, trial, "trial_id, frame")


def contacts(subject=None, condition=None, trial=None):
    """One row per gait contact, sorted by trial and chronological contact number.

    Contact numbers start at one in each trial. Zero-based start_frame is
    inclusive and stop_frame is exclusive. Quiet stance has no contact rows.
    """
    _require_resource("contacts", "a gait contacts table")
    return _trial_measurements("contacts", subject, condition, trial, "trial_id, contact")


def contact_image(subject, condition, trial, contact, *, registered=False):
    """Return one gait contact's peak pressures in kPa as a [y, x] array.

    Contact numbers start at one within each trial. The default image uses the
    original platform grid and only sensors on the selected foot during its
    contact interval. For walking, registered=True returns the aligned image
    on the participant's template grid for that condition and side.
    """
    contact = _positive_int(contact, "Contact")
    trial_id, _ = _trial(subject, condition, trial)
    if registered and condition not in ("ws", "wc", "wf"):
        raise LookupError("Registered contact images are available for walking trials")
    _require_resource("contacts", "a gait contacts table")
    if registered:
        _require_resource("footprints", "a registered footprints table")
    name = "footprints" if registered else "pressure"
    with _source.connect(("contacts", name)) as con:
        step = con.execute(
            "SELECT side, start_frame, stop_frame FROM contacts WHERE trial_id = ? AND contact = ?",
            [trial_id, contact],
        ).fetchone()
        if step is None:
            raise LookupError(f"No contact {contact} in {subject} {condition} trial {trial}")
        side, start, stop = step
        if registered:
            cells = con.execute(
                """SELECT y, x, value FROM footprints
                   WHERE subject = ? AND condition = ? AND side = ? AND trial_id = ? AND contact = ?""",
                [subject, condition, side, trial_id, contact],
            ).fetchnumpy()
        else:
            cells = con.execute(
                """SELECT y, x, max(value) AS value FROM pressure
                   WHERE trial_id = ? AND side = ? AND frame >= ? AND frame < ? GROUP BY y, x""",
                [trial_id, side, start, stop],
            ).fetchnumpy()
    if not len(cells["value"]):
        raise LookupError(f"No {name} pixels for {subject} {condition} trial {trial}, contact {contact}")
    image = np.zeros((NROWS, NCOLS), dtype=cells["value"].dtype)
    image[cells["y"], cells["x"]] = cells["value"]
    return image


def footprints(subject, condition, side):
    """Registered maps [footprint, y, x] in kPa for one subject, task and foot. Index 0 is the template."""
    if not isinstance(subject, str) or not isinstance(condition, str):
        raise TypeError("Footprints require one participant and one condition")
    side = _side_code(side)
    _require_resource("footprints", "a registered footprints table")
    with _source.connect("footprints") as con:
        cells = _select_rows(con, "footprints", subject=subject, condition=condition, side=side)
    grid = np.zeros((int(cells.footprint.max()) + 1 if len(cells) else 0, NROWS, NCOLS))
    grid[cells.footprint.to_numpy(), cells.y.to_numpy(), cells.x.to_numpy()] = cells.value
    return grid


def scan(subject, side):
    """Return one foot and lower-leg mesh as (vertices in meters, triangular faces).

    Local extracted STLs are read directly. For a packaged release, only the named
    member of scans.zip is read, using byte ranges for HTTP.
    """
    if not isinstance(subject, str) or not re.fullmatch(r"s[0-9]{3}", subject):
        raise ValueError("Expected a participant ID (sNNN)")
    member = f"scans/{subject}{_side_code(side).lower()}_foot.stl"
    if isinstance(_source.root, Path) and Path(location(member)).is_file():
        raw = Path(location(member)).read_bytes()
    else:
        _require_resource("scans", "a foot scans archive")
        with _source.open_resource("scans") as stream, zipfile.ZipFile(stream) as archive:
            raw = archive.read(member)
    rec = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * rec.itemsize:
        raise ValueError(f"Invalid binary STL length for {member}")
    triangles = np.frombuffer(raw, rec, count=count, offset=84)["v"].reshape(-1, 3)
    vertices, inverse = np.unique(triangles, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def connect():
    """Open an independent DuckDB session over all PPDS tables.

    The subjects view projects validated participant descriptors from trials. The
    pressure, pressure_features and gait contacts views include trial_id, subject,
    condition and trial for joins. The standing view contains sparse quiet-stance
    pressure cells. Releases predating the contacts table omit its view.
    """
    resources = _source.metadata["resources"]
    con = _source.connect(("trials", "footprints") if "footprints" in resources else "trials")
    try:
        _subject_view(con, _source.metadata["subject_columns"])
        _with_trial_keys(con, "pressure")
        for name in ("pressure_features", "contacts"):
            if name in resources:
                _with_trial_keys(con, name)
        con.execute("CREATE VIEW standing AS SELECT * FROM pressure WHERE condition = 'ss'")
    except BaseException:
        con.close()
        raise
    return con
