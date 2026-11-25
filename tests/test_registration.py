"""Test footprint geometry and Zebris trial boundary cleanup."""

import numpy as np
import pytest

from ppds import registration as reg
from ppds import zebris


def test_boundary_cleanup_removes_entry_foot_and_keeps_axis_order():
    data = np.zeros((8, 100, 12))  # original ASCII axes, before quarter-turn
    data[2:5, 2:7, :3] = 2  # already loaded at start: partial entry foot
    data[2:5, 50:57, 5:9] = 4  # complete later foot, well beyond margin
    cleaned, warnings = zebris.clean_zebris_trial(data, trial=7)
    expected = np.rot90(data[:, :, 4:10], k=3).transpose(2, 0, 1)
    expected[expected == 2] = 0
    np.testing.assert_array_equal(cleaned, expected)
    assert warnings == [(7, "Trial started with subject already on the platform.")]


def test_boundary_cleanup_empty_trial_is_empty():
    cleaned, warnings = zebris.clean_zebris_trial(np.zeros((8, 100, 4)))
    assert cleaned.shape == (0, 100, 8)
    assert warnings == []


@pytest.mark.parametrize(("columns", "frames"), [
    (slice(10, 13), slice(2, 3)),  # final-step margin crosses column zero
    (slice(50, None), slice(0, 1)),  # entry step has no following gap
    (slice(0, 30), slice(2, 3)),  # exit step has no preceding gap
    (slice(None), slice(1, 2)),  # border contact spans the whole walkway
    (slice(40, 45), slice(None)),  # the same partial step spans both time boundaries
])
def test_boundary_cleanup_discards_isolated_partial_contact(columns, frames):
    data = np.zeros((8, 100, 3))
    data[2, columns, frames] = 1
    cleaned, warnings = zebris.clean_zebris_trial(data)
    assert warnings
    assert cleaned.shape == (0, 100, 8)


@pytest.mark.parametrize("closing", [0, 1])
def test_segmentation_threshold_controls_connected_components_and_output(closing):
    image = np.zeros((18, 32))
    image[4:11, 4:10] = 9
    image[6, 6] = 4  # closing can fill this hole for connectivity
    image[5:10, 18:24] = 2
    (foot,) = reg.segment(image, threshold=4, closing=closing)
    np.testing.assert_array_equal(foot, np.where(image > 4, image, 0))


def test_moments_use_only_coordinates_with_weight_above_threshold():
    image = np.array([[1, 10, 0], [0, 0, 20], [0, 0, 4]])
    np.testing.assert_allclose(reg.centroid(image, 4), (5 / 3, 2 / 3))
    assert reg.principal_axis(image, 4) == pytest.approx(np.pi / 4)
    with pytest.raises(ValueError, match="threshold"):
        reg.centroid(image, 20)


def ellipse(xc, yc, degrees):
    y, x = np.indices((110, 110))
    angle = np.deg2rad(degrees)
    along = (x - xc) * np.cos(angle) + (y - yc) * np.sin(angle)
    across = -(x - xc) * np.sin(angle) + (y - yc) * np.cos(angle)
    support = (along / 16) ** 2 + (across / 4) ** 2 < 1
    # A stronger positive end makes a half-turn visible even on an undirected axis.
    return np.where(support, 10 + along / 2, 0)


@pytest.mark.parametrize("degrees", [-89, -65, -44, -18, 0, 18, 44, 65, 89, 90])
def test_major_axis_measured_from_the_positive_x_axis(degrees):
    image = ellipse(55, 55, degrees)
    actual = np.rad2deg(reg.principal_axis(image))
    assert -90 <= actual <= 90
    error = (actual - degrees + 90) % 180 - 90
    assert abs(error) < 1.5


@pytest.mark.parametrize("source_angle,target_angle", [(10, 70), (44, 46), (88, 92)])
@pytest.mark.parametrize("binary", [False, True])
def test_registration_aligns_off_center_images_and_keeps_matching_ends(
    source_angle, target_angle, binary
):
    source = ellipse(23, 22, source_angle)
    target = ellipse(72, 70, target_angle)
    output = reg.register_pa(target, source, binary=binary)

    # Independently measure the result in the sensor grid.
    def moments(image):
        y, x = np.nonzero(image)
        w = image[y, x]
        center = np.average([x, y], axis=1, weights=w)
        cov = np.cov([x, y], aweights=w, ddof=0)
        direction = np.linalg.eigh(cov)[1][:, -1]
        return center, direction

    expected_center, expected_axis = moments(target)
    actual_center, actual_axis = moments(output)
    np.testing.assert_allclose(actual_center, expected_center, atol=0.2)
    assert abs(actual_axis @ expected_axis) > np.cos(np.deg2rad(1))
    # High-pressure end of the source stays on the corresponding end of the target.
    expected_end, _ = moments((target > 15).astype(float))
    actual_end, _ = moments((output > 15).astype(float))
    assert (actual_end - actual_center) @ (expected_end - expected_center) > 0


@pytest.mark.parametrize("threshold", [-1, 0])
@pytest.mark.parametrize(("minimum", "sizes"), [(1, [1, 2]), (2, [2]), (3, [])])
def test_segment_minimum_counts_original_active_pixels(threshold, minimum, sizes):
    image = np.full((8, 9), -2)
    image[2, 2] = 5
    image[4, 5] = image[5, 6] = 7
    feet = reg.segment(image, threshold=threshold, n_min=minimum)
    assert [np.count_nonzero(foot) for foot in feet] == sizes


@pytest.mark.parametrize(("closing", "components"), [(0, 2), (1, 2), (2, 1), (True, 1)])
def test_segment_closing_iterations(closing, components):
    image = np.zeros((16, 24))
    image[5:10, 4:8] = 5
    image[5:10, 11:15] = 6
    feet = reg.segment(image, closing=closing)
    assert len(feet) == components
    np.testing.assert_array_equal(np.sum(feet, axis=0), image)


@pytest.mark.parametrize("closing", [0, 1, 2, True])
def test_segment_closing_keeps_footprints_on_the_image_border(closing):
    image = np.zeros((12, 14))
    image[0:4, 0:3] = 5
    image[0:4, 9:14] = 3
    image[11, 6] = 1
    feet = reg.segment(image, closing=closing)
    np.testing.assert_array_equal(np.sum(feet, axis=0), image)
    assert len(feet) == 3


@pytest.mark.parametrize(("closing", "error"), [(-1, ValueError), (0.5, TypeError)])
def test_segment_requires_nonnegative_integer_closing(closing, error):
    with pytest.raises(error):
        reg.segment(np.ones((5, 5)), closing=closing)
