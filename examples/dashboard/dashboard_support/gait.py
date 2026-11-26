"""Contact-centred plantar COP paths and bilateral load transfer for the gait report."""

from dataclasses import dataclass

import numpy as np
from dashboard_support.measurements import SIDES

from ppds.dataset import CELL_CM


@dataclass
class Footprint:
    image: np.ndarray
    extent: np.ndarray  # sensor edges, in pitches relative to the contact centre
    offset: float  # lateral translation into the two-foot display


@dataclass
class CopDiagram:
    feet: dict[str, Footprint]
    paths: dict[tuple[int, int], np.ndarray]  # (trial, contact) in the local footprint
    sides: dict[tuple[int, int], str]
    butterfly: dict[int, np.ndarray]  # one bilateral path per pass, with gaps


def _footprints(cells, contacts):
    """Translate pressure maps on the sensor grid; retain their physical scale and orientation."""
    centres, feet = {}, {}
    pixels = cells.merge(
        contacts[["trial", "contact", "side"]],
        on=["trial", "contact"],
        validate="many_to_one",
    )
    for side, group in pixels.groupby("side"):
        positions, pressures = [], []
        for key, step in group.groupby(["trial", "contact"]):
            xy = step[["x", "y"]].to_numpy(dtype=int)
            peak = step.peak_pressure.to_numpy()
            centre = np.rint(np.average(xy, axis=0, weights=peak)).astype(int)
            centres[key] = centre
            positions.append(xy - centre)
            pressures.append(peak)
        xy = np.concatenate(positions)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        image = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
        np.add.at(image, (xy[:, 1] - y0, xy[:, 0] - x0), np.concatenate(pressures))
        extent = np.array([x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5])
        # A fixed five-sensor gap separates the two schematic footprints.
        offset = -2.5 - extent[1] if side == "L" else 2.5 - extent[0]
        feet[side] = Footprint(image / len(positions), extent, offset)
    return centres, feet


def cop_diagram(cells, contacts, frames, *, walking):
    """Apply the same translations to pressure and COP, then combine simultaneous foot loads.

    ``walking`` passes are trimmed to their measured double support (see below);
    running passes keep every loaded sample.
    """
    centres, feet = _footprints(cells, contacts)
    paths, sides, butterfly = {}, {}, {}
    for trial, trace in frames.groupby("trial"):
        trace = trace.sort_values("frame")
        weighted = np.zeros((len(trace), 2))
        load = np.zeros(len(trace))
        support = np.zeros((len(trace), 2), dtype=int)
        for step in contacts.loc[contacts.trial.eq(trial)].itertuples():
            key = (trial, step.contact)
            side = step.side
            suffix = SIDES[side].lower()
            interval = slice(step.start_frame, step.stop_frame)
            stance = trace.iloc[interval]
            xy = stance[[f"cop_x_{suffix}", f"cop_y_{suffix}"]].to_numpy() / (
                CELL_CM * 10
            )
            local = xy - centres[key]
            paths[key] = local
            sides[key] = side
            force = stance[f"grf_z_{suffix}"].to_numpy()
            loaded = force > 0
            indices = np.flatnonzero(loaded) + step.start_frame
            weighted[indices] += (local[loaded] + [feet[side].offset, 0]) * force[
                loaded, None
            ]
            load[indices] += force[loaded]
            support[indices, 0 if side == "L" else 1] = step.contact

        # The first and last contacts of a walking pass share double support with a
        # foot off the walkway, which the plate does not measure, so the path would
        # start at an unpaired heel and end at an unpaired toe, off the loop. A
        # walking pass is drawn from its first to its last measured double support.
        # Running has no double support; its brief two-foot overlaps must not trim.
        both = np.flatnonzero((support != 0).all(axis=1))
        if walking and both.size:
            span = slice(both[0], both[-1] + 1)
            weighted, load, support = weighted[span], load[span], support[span]

        combined = np.full_like(weighted, np.nan)
        np.divide(weighted, load[:, None], out=combined, where=load[:, None] > 0)
        # Even adjacent loaded samples can belong to disjoint contacts. Draw a transfer
        # only when a contact is shared across the two samples (including double support).
        shared = ((support[1:] == support[:-1]) & (support[1:] != 0)).any(axis=1)
        switches = ~shared & (load[:-1] > 0) & (load[1:] > 0)
        butterfly[trial] = np.insert(
            combined, np.flatnonzero(switches) + 1, np.nan, axis=0
        )
    return CopDiagram(feet, paths, sides, butterfly)
