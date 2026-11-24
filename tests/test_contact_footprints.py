"""Walking footprint identity, pressure units, registration and template boundaries."""

from math import cos, sin

import numpy as np
import pyarrow as pa
import pytest
from scipy import ndimage

from ppds import contact_geometry as geometry
from ppds import registration
from ppds.contact_geometry import GRID_SHAPE, PITCH_MM


@pytest.mark.parametrize("dtype", [np.uint16, np.float64])
@pytest.mark.parametrize(
    "transform",
    [
        (0, 0, 0),
        (2.25, -7.1, 0),
        (0.33, 4.02, -0.28),
        np.array([0.33, 4.02, -0.28], dtype=np.float32),
    ],
)
def test_rigid_transform_matches_original_scalar_interpolation(dtype, transform):
    image = np.random.default_rng(765).integers(0, 550, size=(48, 21)).astype(dtype)
    center = (10.25, 24.15)
    dx, dy, theta = transform
    cy, cx = center[::-1]
    co, si = cos(theta), sin(theta)

    def scalar_coordinate(position):
        y, x = position
        return (
            cy + (-dy) * co - (-dx) * si + y * co - x * si - cy * co + cx * si,
            cx + (-dy) * si + (-dx) * co + y * si + x * co - cy * si - cx * co,
        )

    legacy = ndimage.geometric_transform(image, scalar_coordinate, order=1)
    np.testing.assert_array_equal(
        registration.transformXYR2D(image, transform, center), legacy
    )


def test_published_footprint_replays_archived_grid_center_interpolation():
    template = np.zeros((25, 25))
    template[4:13, 5:9] = 8
    template[4:7, 4:7] = 50
    source = np.zeros_like(template)
    source[8:17, 7:11] = 6
    source[15:20, 10:13] = 18

    image = geometry.register_contact_peak(source, geometry.register_contact_peak(template))
    # Reference pressures from the original scalar implementation.
    assert np.count_nonzero(image) == 90
    np.testing.assert_allclose(
        image[[8, 16, 17], [12, 14, 12]],
        [1.419814659805553, 13.686848221586825, 5.579404354325959],
        rtol=1e-12,
    )


def trial_data(trial_id, origin, value):
    # The right foot is unloaded for frame 2; both feet are loaded in frame 3.
    rows = []
    records = []
    for contact, (side, frames, dy, dx) in enumerate(
        [("R", (1, 3), 0, 10), ("L", (3, 4), 30, 0), ("R", (5, 6), 60, 14)], 1
    ):
        foot = [
            (origin + dy + y, dx + x, value + contact * x + y * 3)
            for y, x in [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)]
        ]
        for i, frame in enumerate(frames):
            for y, x, kpa in foot:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "frame": frame,
                        "y": y,
                        "x": x,
                        "value": kpa + i,
                        "side": side,
                    }
                )
        records.append(
            {
                "trial_id": trial_id,
                "contact": contact,
                "side": side,
                "start_frame": frames[0],
                "stop_frame": frames[-1] + 1,
                "peak_pressure": max(k for _, _, k in foot) + 1,
                "contact_y_min": foot[0][0] * PITCH_MM,
                "contact_y_max": foot[-1][0] * PITCH_MM,
            }
        )
    return sorted(rows, key=lambda r: (r["frame"], r["y"], r["x"])), records


def sensor_map(rows, record):
    image = np.zeros(GRID_SHAPE)
    for row in rows:
        if (
            row["side"] == record["side"]
            and record["start_frame"] <= row["frame"] < record["stop_frame"]
        ):
            y, x = row["y"], row["x"]
            image[y, x] = max(image[y, x], row["value"])
    return image


def test_original_pressure_contact_keeps_gap_double_support_and_cellwise_maximum():
    rows, contacts = trial_data(7, 10, 20)
    peaks = geometry.original_contact_peaks(
        pa.Table.from_pylist(rows), pa.Table.from_pylist(contacts)
    )
    assert len(peaks) == 3
    for record, image in peaks:
        np.testing.assert_array_equal(image, sensor_map(rows, record))

    with pytest.raises(ValueError, match="cover"):
        geometry.original_contact_peaks(pa.Table.from_pylist(rows), contacts[:-1])
    with pytest.raises(ValueError, match="overlap"):
        geometry.original_contact_peaks(
            pa.Table.from_pylist(rows), [contacts[0] | {"stop_frame": 6}, *contacts[1:]]
        )
    with pytest.raises(ValueError, match="pressure or sensor bounds"):
        geometry.original_contact_peaks(
            pa.Table.from_pylist(rows),
            [contacts[0] | {"contact_y_min": 0.0}, *contacts[1:]],
        )
