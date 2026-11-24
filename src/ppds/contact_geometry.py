"""Original and registered peak-pressure images for an individual contact."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import registration
from .dataset import CELL_CM, NCOLS, NROWS

GRID_SHAPE = (NROWS, NCOLS)
PITCH_MM = CELL_CM * 10


def original_contact_peaks(pressure, contacts):
    """Return (contact record, original-grid peak kPa image) pairs for one walking trial.

    Supply the trial's complete Arrow pressure table and all its contact records
    as an Arrow table or an iterable of mappings.
    A cell belongs to the contact of its approved side in [start_frame, stop_frame).
    A brief unloaded gap within that interval leaves the contact intact.
    """
    if isinstance(contacts, pa.Table):
        contacts = contacts.to_pylist()
    records = sorted(contacts, key=lambda r: r["contact"])
    if not records or not len(pressure):
        raise ValueError("Expected walking contacts and positive pressure samples")
    trial_id = records[0]["trial_id"]
    if any(
        pressure[k].null_count for k in ("trial_id", "frame", "y", "x", "value", "side")
    ):
        raise ValueError(
            f"Missing pressure coordinates, value or side for trial {trial_id}"
        )
    if pc.unique(pressure["trial_id"]).to_pylist() != [trial_id]:
        raise ValueError("Walking pressure must belong to the contacts' trial")
    frame, y, x, values = (pressure[k].to_numpy() for k in ("frame", "y", "x", "value"))
    if (
        values.min() <= 0
        or y.min() < 0
        or y.max() >= GRID_SHAPE[0]
        or x.min() < 0
        or x.max() >= GRID_SHAPE[1]
    ):
        raise ValueError(
            f"Invalid walking pressure coordinates or values for trial {trial_id}"
        )

    side_masks = {
        side: pc.equal(pressure["side"], side).to_numpy() for side in ("L", "R")
    }
    last_stop = {"L": 0, "R": 0}
    last_start = -1
    for number, record in enumerate(records, 1):
        side, start, stop = (record[k] for k in ("side", "start_frame", "stop_frame"))
        if (
            record["trial_id"] != trial_id
            or record["contact"] != number
            or side not in last_stop
            or not last_start < start < stop
        ):
            raise ValueError(
                f"Invalid chronological contact {number} for trial {trial_id}"
            )
        if start < last_stop[side]:
            raise ValueError(f"Same-side contacts overlap for trial {trial_id}")
        last_start, last_stop[side] = start, stop

    peaks = []
    assigned = 0
    for record in records:
        start, stop = record["start_frame"], record["stop_frame"]
        mask = (frame >= start) & (frame < stop) & side_masks[record["side"]]
        f, rows, cols, kpa = (a[mask] for a in (frame, y, x, values))
        if not len(f) or f.min() != start or int(f.max()) + 1 != stop:
            raise ValueError(
                f"Pressure differs from the recorded contact interval: {record}"
            )
        if (
            kpa.max() != record["peak_pressure"]
            or int(rows.min()) * PITCH_MM != record["contact_y_min"]
            or int(rows.max()) * PITCH_MM != record["contact_y_max"]
        ):
            raise ValueError(
                f"Original peak pressure or sensor bounds differ for contact: {record}"
            )
        peak = np.zeros(GRID_SHAPE, dtype=np.float64)
        np.maximum.at(peak, (rows, cols), kpa)
        peaks.append((record, peak))
        assigned += len(f)
    if assigned != len(pressure):
        raise ValueError(
            f"Walking contacts do not cover every pressure sample in trial {trial_id}"
        )
    return peaks


def register_contact_peak(peak, template=None):
    """Pressure-center the original image, then align its binary shape to the centered template.

    With no template, the centered image establishes footprint 0 for one subject,
    walking condition and anatomical side. Use floating-point kPa throughout both
    linear interpolations. This reproduces the published PPDS convention using
    the grid-center pivot on pre-centered images.
    """
    centered = registration.center(np.asarray(peak, dtype=np.float64))
    if template is None:
        return centered
    return registration.register_published_footprint(template, centered)
