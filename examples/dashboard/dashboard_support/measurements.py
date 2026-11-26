"""Prepare the selected trials' PPDS measurements for the dashboard."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from reader_data import Release

from ppds import dataset as ds
from ppds.paths import local_data_dir
from ppds.release_layout import REFERENCE

CONDITIONS = {
    "Comfortable walking": "wc",
    "Slow walking": "ws",
    "Fast walking": "wf",
    "Comfortable running": "rc",
    "Quiet standing": "ss",
}
SIDES = {"L": "Left", "R": "Right"}
CONTACT_COLUMNS = {
    "contact": "Contact",
    "side": "Foot",
    "contact_time": "Duration (s)",
    "peak_pressure": "Peak pressure (kPa)",
    "peak_grf_z": "Peak vertical force (N)",
    "vertical_impulse": "Vertical impulse (N·s)",
    "max_contact_area": "Max. area (cm²)",
    "strike_index": "Strike index",
}

CONTACT_PRECISION = {
    "contact_time": 2,
    "peak_pressure": 1,
    "peak_grf_z": 1,
    "vertical_impulse": 1,
    "max_contact_area": 1,
    "strike_index": 3,
}


def contact_details(contacts):
    rows = contacts[list(CONTACT_COLUMNS)].round(CONTACT_PRECISION)
    rows["side"] = rows.side.map(SIDES)
    return rows.rename(columns=CONTACT_COLUMNS)


def contact_summary(contacts):
    """Pool individual contacts across the selected passes, giving each contact equal weight."""
    rows = []
    for side, label in SIDES.items():
        steps = contacts.loc[contacts.side.eq(side)]
        row = {"Foot": label, "Trials": steps.trial.nunique(), "Contacts": len(steps)}
        for column, places in CONTACT_PRECISION.items():
            values = steps[column].dropna()
            stat = f"{values.mean():.{places}f}" if len(values) else "—"
            if len(values) > 1:
                stat += f" ± {values.std():.{places}f}"
            row[CONTACT_COLUMNS[column]] = stat
        rows.append(row)
    return pd.DataFrame(rows)


def open_release():
    """Prefer an explicit download, then the complete manuscript cache, then Zenodo."""
    local = local_data_dir()
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "ppds"
        / str(ds.RECORD)
    )
    if os.environ.get("PPDS_DATA", "").strip() or (local / "metadata.json").is_file():
        ds.use_local(local)
        return Release.local(local)
    elif (cache / "metadata.json").is_file() and all(
        (cache / spec["zenodo_filename"]).is_file()
        for spec in REFERENCE["resources"].values()
    ):
        ds.use_local(cache)
        return Release.local(cache)
    else:
        ds.use_zenodo(ds.RECORD)
        return Release.zenodo(ds.RECORD)


def _trial_rows(con, trials, table, columns, order):
    # Filter the physical Parquet trial IDs before joining the displayed trial
    # numbers. This lets browser clients fetch just the selected row groups.
    # The IDs are inlined as integer literals: with bound parameters, DuckDB
    # (1.5 in Pyodide, 2.0 natively) skips row-group pruning and reads the whole
    # table, about 50 MB over HTTP for pressure_features.
    ids = ", ".join(str(int(n)) for n in trials.trial_id)
    rows = con.execute(
        f"SELECT {columns} FROM {table} WHERE trial_id IN ({ids}) ORDER BY {order}"
    ).df()
    return rows.merge(
        trials[["trial_id", "trial"]], on="trial_id", validate="many_to_one"
    )


def trial_features(trials, release):
    with release.connect(("pressure_features", "contacts")) as con:
        frames = _trial_rows(con, trials, "pressure_features", "*", "trial_id, frame")
        contacts = _trial_rows(con, trials, "contacts", "*", "trial_id, contact")
    return frames, contacts


def gait_maps(trials, release):
    """Read original contact pixels and registered walking pixels for exactly these passes."""
    ids = trials.trial_id.tolist()
    walking = trials.condition.iloc[0] in ("ws", "wc", "wf")
    tables = ("contact_maps", "footprints") if walking else "contact_maps"
    with release.connect(tables) as con:
        cells = _trial_rows(
            con,
            trials,
            "contact_maps",
            "trial_id, contact, y, x, peak_pressure",
            "trial_id, contact",
        )
        registered = pd.DataFrame()
        if walking:
            registered = con.execute(
                """SELECT trial_id, contact, side, y, x, value FROM footprints
                   WHERE subject = ? AND condition = ? AND trial_id IN (SELECT unnest(?))""",
                [trials.subject.iloc[0], trials.condition.iloc[0], ids],
            ).df()
    peaks = {n: np.zeros((ds.NROWS, ds.NCOLS), dtype=np.uint16) for n in trials.trial}
    for number, pixels in cells.groupby("trial"):
        np.maximum.at(peaks[number], (pixels.y, pixels.x), pixels.peak_pressure)
    return peaks, cells, registered


def mean_footprints(registered, contacts):
    """Use all contacts of a side as the denominator, including zero pressure at unloaded pixels."""
    means = {}
    if not registered.empty:
        for side, pixels in registered.groupby("side"):
            grid = np.zeros((ds.NROWS, ds.NCOLS))
            np.add.at(grid, (pixels.y, pixels.x), pixels.value)
            means[side] = grid / contacts.side.eq(side).sum()
    return means


def stride_curves(frames, contacts):
    """Return time-normalized stance force and stance percent for each complete within-pass stride."""
    result = {side: [] for side in SIDES}
    percent = np.arange(101)
    for trial_id, steps in contacts.groupby("trial_id"):
        trace = frames.loc[frames.trial_id.eq(trial_id)].sort_values("frame")
        forces = {
            side: trace[f"grf_z_{name.lower()}"].to_numpy()
            for side, name in SIDES.items()
        }
        events = list(steps.sort_values("start_frame").itertuples())
        for start, other, end in zip(events, events[1:], events[2:]):
            if start.side == other.side or start.side != end.side:
                continue
            lo, hi = start.start_frame, end.start_frame
            stop = start.stop_frame  # the first sample after this contact
            phase = 100 * np.arange(stop - lo + 1) / (stop - lo)
            curve = np.interp(percent, phase, forces[start.side][lo : stop + 1])
            toe_off = 100 * (stop - lo) / (hi - lo)
            result[start.side].append((curve, toe_off))
    return result


def trial_summary(trials, frames, contacts):
    summary = (
        trials[["trial", "n_frames", "frequency_hz", "duration_s"]]
        .set_index("trial")
        .copy()
    )
    if trials.condition.iloc[0] != "ss":
        for side, label in SIDES.items():
            counts = contacts.loc[contacts.side.eq(side)].groupby("trial").size()
            summary[f"{label} contacts"] = counts.reindex(summary.index, fill_value=0)
    summary["Peak pressure (kPa)"] = frames.groupby("trial").peak_pressure.max()
    summary["Peak combined force (N)"] = frames.groupby("trial").grf_z.max().round(1)
    return summary.reset_index().rename(
        columns={
            "trial": "Trial",
            "n_frames": "Frames",
            "frequency_hz": "Rate (Hz)",
            "duration_s": "Duration (s)",
        }
    )


def standing_summary(frames):
    rows = []
    for label, suffix in [("Left", "_left"), ("Right", "_right"), ("Both feet", "")]:
        rows.append(
            {
                "Support": label,
                "Mean vertical force (N)": frames[f"grf_z{suffix}"].mean(),
                "Mean contact area (cm²)": frames[f"contact_area{suffix}"].mean(),
                "Peak pressure (kPa)": frames[f"peak_pressure{suffix}"].max(),
            }
        )
    return pd.DataFrame(rows).round(1)
