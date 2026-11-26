"""Pressure maps and strike index on small, hand-checked pressure series."""

import numpy as np
import pytest

from ppds import maps


def series():
    # Three frames of a five-row foot: heel first, then flat foot, then toe-off with an unloaded heel.
    frames = np.zeros((3, 6, 2), dtype=np.uint16)
    frames[0, 1, 0] = 10
    frames[1, 1:5, 0] = [30, 10, 10, 20]
    frames[2, 4:6, 0] = [40, 5]
    return frames


def test_peak_integral_and_mean_include_unloaded_frames():
    frames = series()
    np.testing.assert_array_equal(maps.peak_pressure_map(frames)[:, 0], [0, 30, 10, 10, 40, 5])
    assert maps.peak_pressure_map(frames).dtype == np.uint16
    pti = maps.pressure_time_integral_map(frames, 100)
    np.testing.assert_allclose(pti[:, 0], [0, 0.4, 0.1, 0.1, 0.6, 0.05])
    # The mean over the 0.03 s interval counts each sensor's unloaded frames as zero.
    np.testing.assert_allclose(maps.mean_pressure_map(frames), pti / 0.03)


def test_integral_does_not_overflow_integer_pressures():
    frames = np.full((3, 1, 1), 60000, dtype=np.uint16)
    assert maps.pressure_time_integral_map(frames, 10)[0, 0] == pytest.approx(18000)


def test_strike_index_uses_the_first_frame_and_the_whole_footprint():
    # Rows 1 to 5 span sensor edges 0.5 to 5.5; the first-frame center at row 1 lies 0.5 rows in.
    assert maps.strike_index(series(), 0) == pytest.approx(0.1)
    toe_first = series()[::-1]
    # Starting at toe-off, the first-frame center lies at row (40 * 4 + 5 * 5) / 45.
    assert maps.strike_index(toe_first, 0) == pytest.approx((185 / 45 - 0.5) / 5)


def test_strike_index_onset_skips_frames_below_the_force_threshold():
    frames = series()
    # Frame forces: 10, 70 and 45 kPa summed, times 1000 sensor area (unit area here).
    assert maps.strike_index(frames, 50, sensor_area_m2=1e-3) == pytest.approx(
        ((30 * 1 + 10 * 2 + 10 * 3 + 20 * 4) / 70 - 0.5) / 5)
    assert maps.strike_index(frames, 10, sensor_area_m2=1e-3) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="onset force"):
        maps.strike_index(frames, 71, sensor_area_m2=1e-3)
    with pytest.raises(ValueError, match="onset force"):
        maps.strike_index(frames, -1)
    # The PPDS default of 20 N with 8.469 mm sensors needs 279 kPa summed over a frame.
    assert maps.STRIKE_ONSET_N == 20 and 278 < 20 / (1000 * maps.SENSOR_AREA_M2) < 279


def test_strike_index_of_a_single_row_is_its_center():
    frames = np.zeros((2, 4, 3))
    frames[:, 2, 1] = 7
    assert maps.strike_index(frames, 0) == 0.5


@pytest.mark.parametrize("frames", [np.zeros((0, 2, 2)), np.zeros((2, 2)), np.zeros((1, 2, 2), dtype=bool)])
def test_invalid_series_are_rejected(frames):
    with pytest.raises((ValueError, TypeError)):
        maps.peak_pressure_map(frames)


def test_strike_index_requires_a_loaded_first_frame_and_nonnegative_pressure():
    frames = series()
    # An unloaded leading frame is skipped even without a force threshold.
    assert maps.strike_index(np.concatenate([np.zeros_like(frames[:1]), frames]), 0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="onset force"):
        maps.strike_index(np.zeros_like(frames), 0)
    signed = frames.astype(float)
    signed[1, 0, 1] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        maps.strike_index(signed)


@pytest.mark.parametrize("hz", [0, -10, np.nan, np.inf])
def test_integral_requires_a_positive_frequency(hz):
    with pytest.raises(ValueError, match="frequency"):
        maps.pressure_time_integral_map(series(), hz)
