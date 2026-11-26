"""Check dashboard selection scope against PPDS 1.1.1 measurements."""

import re
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest
from reader_data import Release

from ppds import dataset as ds
from ppds.paths import local_data_dir


def test_browser_html_contains_current_dashboard():
    from export_html import DASHBOARD, browser_notebook

    page = (DASHBOARD / "dashboard.html").read_text(encoding="utf-8")
    saved = re.search(r"<marimo-code\b[^>]*>(.*?)</marimo-code>", page, re.DOTALL)
    assert saved is not None
    # Marimo normalizes the script metadata. Compare the notebook and its bundled helpers.
    code = unquote(saved[1]).split("# ///\n", 1)[1]
    assert code == browser_notebook().split("# ///\n", 1)[1], "Rebuild dashboard.html with export_html.py"


@pytest.fixture
def measurements():
    from dashboard_support import measurements

    root = local_data_dir()
    release_files = ("metadata.json", "trials.parquet", "pressure.parquet", "scans.zip")
    if not any((root / name).exists() for name in release_files):
        pytest.skip("requires a prepared PPDS release (set PPDS_DATA to its directory)")
    ds.use_local(root)
    return measurements


@pytest.fixture
def release(measurements):
    return Release.local(local_data_dir())


@pytest.mark.parametrize("passes", [[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
def test_walking_scope(measurements, release, passes):
    trials = ds.trials("s001", "wc")
    selected = trials.loc[trials.trial.isin(passes)]
    contacts = ds.contacts("s001", "wc", passes)
    frames, steps = measurements.trial_features(selected, release)
    expected_frames = ds.pressure_features("s001", "wc", passes)
    pd.testing.assert_frame_equal(frames, expected_frames[frames.columns])
    pd.testing.assert_frame_equal(steps, contacts[steps.columns])
    peaks, cells, registered = measurements.gait_maps(selected, release)
    assert list(peaks) == passes
    assert set(cells.trial) == set(passes)
    for number in passes:
        np.testing.assert_array_equal(
            peaks[number], ds.peak_image("s001", "wc", number)
        )
    means = measurements.mean_footprints(registered, contacts)
    for side in ("L", "R"):
        if len(passes) == 1:
            expected = [
                ds.contact_image("s001", "wc", 1, step.contact, registered=True)
                for step in contacts.loc[contacts.side.eq(side)].itertuples()
            ]
        else:
            expected = ds.footprints("s001", "wc", side)
        np.testing.assert_allclose(means[side], np.mean(expected, axis=0))
    curves = measurements.stride_curves(
        ds.pressure_features("s001", "wc", passes), contacts
    )
    # In this example each of the ten passes contains one complete stride on each side.
    assert {side: len(items) for side, items in curves.items()} == {
        "L": len(passes),
        "R": len(passes),
    }
    assert all(curve.shape == (101,) for items in curves.values() for curve, _ in items)


def test_running_pass_without_a_complete_stride(measurements, release):
    trials = ds.trials("s014", "rc")
    selected = trials.loc[trials.trial.eq(1)]
    contacts = ds.contacts("s014", "rc", 1)
    frames = ds.pressure_features("s014", "rc", 1)
    peaks, cells, registered = measurements.gait_maps(selected, release)
    assert registered.empty
    assert not cells.empty
    np.testing.assert_array_equal(peaks[1], ds.peak_image("s014", "rc", 1))
    assert measurements.stride_curves(frames, contacts) == {"L": [], "R": []}


def test_standing_summary_uses_each_support_signal(measurements):
    trials = ds.trials("s001", "ss")
    frames = ds.pressure_features("s001", "ss", 1)
    contacts = ds.contacts("s001", "ss", 1)
    summary = measurements.trial_summary(trials, frames, contacts)
    assert summary["Duration (s)"].item() == 10
    assert "Left contacts" not in summary
    metrics = measurements.standing_summary(frames).set_index("Support")
    assert metrics.loc["Both feet", "Mean vertical force (N)"] == round(
        frames.grf_z.mean(), 1
    )
    assert metrics.loc["Left", "Peak pressure (kPa)"] == frames.peak_pressure_left.max()


def test_contacts_pool_all_passes_and_display_per_foot(measurements):
    contacts = ds.contacts("s001", "wc")
    summary = measurements.contact_summary(contacts).set_index("Foot")
    assert list(summary.index) == ["Left", "Right"]
    for side, name in measurements.SIDES.items():
        steps = contacts.loc[contacts.side.eq(side)]
        assert summary.loc[name, "Contacts"] == len(steps) == 20
        assert summary.loc[name, "Trials"] == steps.trial.nunique() == 10
        pressure = steps.peak_pressure
        assert (
            summary.loc[name, "Peak pressure (kPa)"]
            == f"{pressure.mean():.1f} ± {pressure.std():.1f}"
        )
    one = measurements.contact_summary(contacts.iloc[:1]).set_index("Foot")
    assert one.loc["Left", "Contacts"] == 0
    assert one.loc["Left", "Duration (s)"] == "—"
    assert one.loc["Right", "Duration (s)"] == "0.59"


def test_running_shows_actual_peak_contact_for_each_foot(measurements, release):
    from dashboard_support import plots

    trials = ds.trials("s001", "rc")
    contacts = ds.contacts("s001", "rc")
    _, cells, _ = measurements.gait_maps(trials, release)
    figure = plots.contact_maps(cells, contacts, 1000)
    for side, ax in zip(measurements.SIDES, figure.axes[:2]):
        rows = contacts.loc[contacts.side.eq(side)]
        step = rows.sort_values(
            ["peak_pressure", "trial", "contact"], ascending=[False, True, True]
        ).iloc[0]
        expected = ds.contact_image("s001", "rc", int(step.trial), int(step.contact))
        y, x = np.nonzero(expected)
        np.testing.assert_array_equal(
            ax.images[0].get_array().filled(0),
            expected[y.min() : y.max() + 1, x.min() : x.max() + 1],
        )
        assert f"trial {step.trial}, contact {step.contact}" in ax.get_title()
        assert ax.images[0].get_clim() == (0, 1000)
    single_side = plots.contact_maps(cells, contacts.loc[contacts.side.eq("L")], 1000)
    assert not single_side.axes[1].images


def test_scan_has_two_to_scale_meshes_with_clear_gap_and_transparency(measurements):
    from dashboard_support import plots

    meshes = {side: ds.scan("s001", side) for side in measurements.SIDES}
    fig = plots.plantar_scan(meshes)
    left, right = fig.data
    assert [left.name, right.name] == ["Left", "Right"]
    assert min(right.x) - max(left.x) == pytest.approx(5)
    for side, trace in zip(measurements.SIDES, fig.data):
        vertices, faces = meshes[side]
        actual = np.column_stack([trace.x, trace.y, trace.z])
        np.testing.assert_allclose(
            np.linalg.norm(actual[1:100] - actual[0], axis=1),
            np.linalg.norm(vertices[1:100] - vertices[0], axis=1) * 100,
            atol=1e-4,
        )
        np.testing.assert_array_equal(
            np.column_stack([trace.i, trace.j, trace.k]), faces
        )
    assert fig.layout.paper_bgcolor == fig.layout.scene.bgcolor == "rgba(0,0,0,0)"


@pytest.mark.parametrize("subject", ["s001", "s020"])
def test_standing_map_keeps_sensor_orientation_and_crops_both_feet(
    measurements, subject
):
    from dashboard_support import plots

    mean = ds.standing_image(subject, kind="mean")
    y, x = np.nonzero(mean)
    figure = plots.standing_pressure(mean, 300)
    ax = figure.axes[0]
    image = ax.images[0]
    assert image.origin == "lower"
    np.testing.assert_array_equal(
        image.get_array().filled(0), mean[y.min() : y.max() + 1, x.min() : x.max() + 1]
    )
    pitch = ds.CELL_CM * 10
    np.testing.assert_allclose(
        image.get_extent(),
        np.array([x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5]) * pitch,
    )
    np.testing.assert_allclose(
        ax.get_xlim(), np.array([x.min() - 2.5, x.max() + 2.5]) * pitch
    )
    np.testing.assert_allclose(
        ax.get_ylim(), np.array([y.min() - 2.5, y.max() + 2.5]) * pitch
    )
    assert ax.get_aspect() == 1


def test_standing_grf_and_both_cop_components_are_time_series(measurements):
    from dashboard_support import plots

    trial = ds.trials("s001", "ss").iloc[0]
    frames = ds.pressure_features("s001", "ss", 1)
    force, cop = plots.standing_time_series(trial, frames)
    for ax, column in zip(
        [force.axes[0], *cop.axes], ["grf_z", "cop_x", "cop_y"], strict=True
    ):
        assert len(ax.lines) == 3
        for line, suffix in zip(ax.lines, ["_left", "_right", ""], strict=True):
            np.testing.assert_allclose(
                line.get_xdata(), frames.frame / trial.frequency_hz
            )
            np.testing.assert_allclose(
                line.get_ydata(), frames[column + suffix], equal_nan=True
            )


@pytest.mark.parametrize(
    ("subject", "condition", "passes"),
    [
        ("s001", "wc", [1, 2]),
        ("s001", "rc", [1]),
        ("s014", "rc", [1]),
        # Two frames with both feet loaded: a running overlap, not double support.
        ("s004", "rc", [2]),
    ],
)
def test_gait_report_uses_contact_relative_cop_and_measured_forces(
    measurements, release, subject, condition, passes
):
    from dashboard_support import plots
    from dashboard_support.gait import cop_diagram

    trials = ds.trials(subject, condition).query("trial in @passes")
    frames = ds.pressure_features(subject, condition, passes)
    steps = ds.contacts(subject, condition, passes)
    _, pixels, _ = measurements.gait_maps(trials, release)
    walking = condition != "rc"
    diagram = cop_diagram(pixels, steps, frames, walking=walking)
    curves = measurements.stride_curves(frames, steps)
    fig = plots.gait_report(trials, frames, diagram, curves, passes[-1])
    left, butterfly, right, force, cycle_r, cycle_l = fig.axes
    assert [len(ax.images) for ax in (left, butterfly, right)] == [1, 2, 1]
    assert [len(ax.lines) for ax in (left, right)] == [
        sum(steps.side == side) for side in ("L", "R")
    ]
    assert cycle_r.get_xlim() == cycle_l.get_xlim() == (0, 100)
    offset = 0
    for i, trial in enumerate(trials.itertuples()):
        trace = frames.loc[frames.trial_id.eq(trial.trial_id)].sort_values("frame")
        force_left = [
            line for line in force.lines if line.get_color() == plots.COLORS["L"]
        ][i]
        np.testing.assert_allclose(
            force_left.get_xdata(), offset + trace.frame / trial.frequency_hz
        )
        np.testing.assert_allclose(force_left.get_ydata(), trace.grf_z_left)
        offset += trial.duration_s
        local = {}
        for side, label in measurements.SIDES.items():
            xy = np.full((len(trace), 2), np.nan)
            for step in steps.loc[
                steps.trial.eq(trial.trial) & steps.side.eq(side)
            ].itertuples():
                points = pixels.loc[
                    pixels.trial.eq(trial.trial) & pixels.contact.eq(step.contact)
                ]
                centre = np.rint(
                    np.average(points[["x", "y"]], axis=0, weights=points.peak_pressure)
                )
                interval = slice(step.start_frame, step.stop_frame)
                xy[interval] = (
                    trace[
                        [f"cop_x_{label.lower()}", f"cop_y_{label.lower()}"]
                    ].to_numpy()[interval]
                    / (ds.CELL_CM * 10)
                    - centre
                )
                np.testing.assert_allclose(
                    diagram.paths[(trial.trial, step.contact)],
                    xy[interval],
                    equal_nan=True,
                )
            xy[:, 0] += diagram.feet[side].offset
            local[side] = xy
        weights = trace[["grf_z_left", "grf_z_right"]].to_numpy()
        expected = np.nansum(
            np.stack([local["L"], local["R"]], axis=1) * weights[:, :, None], axis=1
        )
        loaded = weights.sum(axis=1) > 0
        expected[loaded] /= weights.sum(axis=1)[loaded, None]
        expected[~loaded] = np.nan
        # A walking pass runs from its first to its last double support; a
        # running pass keeps every loaded sample, overlaps included.
        both = np.flatnonzero((weights > 0).all(axis=1))
        if walking and both.size:
            span = slice(both[0], both[-1] + 1)
            expected, loaded = expected[span], loaded[span]
        actual = diagram.butterfly[trial.trial]
        np.testing.assert_allclose(actual[np.isfinite(actual[:, 0])], expected[loaded])
        if condition == "rc" and (~loaded).any():
            assert np.isnan(actual).any()
    for ax, side in ((cycle_r, "R"), (cycle_l, "L")):
        if curves[side]:
            assert ax.lines[0].get_xdata()[-1] == pytest.approx(
                np.mean([toe_off for _, toe_off in curves[side]])
            )
        else:
            assert not ax.lines


def test_butterfly_breaks_between_adjacent_disjoint_contacts(measurements):
    from dashboard_support.gait import cop_diagram

    # A right onset immediately follows left toe-off; there is no double support
    # and no unloaded frame between them. A straight crossover would be spurious.
    steps = pd.DataFrame(
        {
            "trial": [1, 1],
            "contact": [1, 2],
            "side": ["L", "R"],
            "start_frame": [0, 2],
            "stop_frame": [2, 4],
        }
    )
    pixels = steps[["trial", "contact"]].assign(
        x=[2, 12], y=[8, 24], peak_pressure=[20, 30]
    )
    frames = pd.DataFrame(
        {
            "trial": [1] * 4,
            "frame": range(4),
            "grf_z_left": [10, 10, 0, 0],
            "grf_z_right": [0, 0, 20, 20],
        }
    )
    for side, x, y in [("left", 2, 8), ("right", 12, 24)]:
        loaded = frames[f"grf_z_{side}"] > 0
        frames[f"cop_x_{side}"] = np.where(loaded, x * ds.CELL_CM * 10, np.nan)
        frames[f"cop_y_{side}"] = np.where(loaded, y * ds.CELL_CM * 10, np.nan)
    combined = cop_diagram(pixels, steps, frames, walking=False).butterfly[1]
    assert len(combined) == 5
    assert np.isnan(combined[2]).all()
    assert np.isfinite(combined[[0, 1, 3, 4]]).all()


def test_butterfly_spans_each_pass_from_first_to_last_double_support(measurements):
    from dashboard_support.gait import cop_diagram

    # Three walking contacts: L1, R1, L2. The pass's first and last contacts share
    # double support with a foot off the walkway, so their unpaired heel (start)
    # and toe (end) would hang off the loop. The path keeps frames 4..9 only.
    steps = pd.DataFrame(
        {
            "trial": [1, 1, 1],
            "contact": [1, 2, 3],
            "side": ["L", "R", "L"],
            "start_frame": [0, 4, 8],
            "stop_frame": [6, 10, 14],
        }
    )
    pixels = steps[["trial", "contact"]].assign(
        x=[2, 12, 2], y=[8, 24, 40], peak_pressure=[20, 30, 20]
    )
    frames = pd.DataFrame({"trial": [1] * 14, "frame": range(14)})
    left = np.zeros(14)
    left[0:6] = 10
    left[8:14] = 10
    right = np.zeros(14)
    right[4:10] = 20
    frames["grf_z_left"] = left
    frames["grf_z_right"] = right
    for side in ("left", "right"):
        loaded = frames[f"grf_z_{side}"] > 0
        # The COP walks heel to toe in each contact: y rises frame by frame.
        frames[f"cop_x_{side}"] = np.where(loaded, 5 * ds.CELL_CM * 10, np.nan)
        frames[f"cop_y_{side}"] = np.where(
            loaded, (frames.frame + 1) * ds.CELL_CM * 10, np.nan
        )
    diagram = cop_diagram(pixels, steps, frames, walking=True)
    combined = diagram.butterfly[1]
    both = (left > 0) & (right > 0)
    assert np.flatnonzero(both).tolist() == [4, 5, 8, 9]
    assert len(combined) == 6
    assert np.isfinite(combined).all()
    # The single-foot paths still hold every stance sample.
    assert [len(path) for path in diagram.paths.values()] == [6, 6, 6]


def test_running_butterfly_keeps_every_loaded_sample_across_an_overlap(measurements):
    from dashboard_support.gait import cop_diagram

    # Running contacts R1, L1 with a two-frame overlap (frames 4 and 5): not
    # double support, so nothing is trimmed.
    steps = pd.DataFrame(
        {
            "trial": [1, 1],
            "contact": [1, 2],
            "side": ["R", "L"],
            "start_frame": [0, 4],
            "stop_frame": [6, 10],
        }
    )
    pixels = steps[["trial", "contact"]].assign(
        x=[12, 2], y=[8, 24], peak_pressure=[30, 20]
    )
    frames = pd.DataFrame({"trial": [1] * 10, "frame": range(10)})
    frames["grf_z_right"] = np.where(frames.frame < 6, 20.0, 0.0)
    frames["grf_z_left"] = np.where(frames.frame >= 4, 10.0, 0.0)
    for side in ("left", "right"):
        loaded = frames[f"grf_z_{side}"] > 0
        frames[f"cop_x_{side}"] = np.where(loaded, 5 * ds.CELL_CM * 10, np.nan)
        frames[f"cop_y_{side}"] = np.where(
            loaded, (frames.frame + 1) * ds.CELL_CM * 10, np.nan
        )
    combined = cop_diagram(pixels, steps, frames, walking=False).butterfly[1]
    assert len(combined) == 10
    assert np.isfinite(combined).all()


@pytest.mark.parametrize("count", [1, 9, 10])
def test_walkway_shared_axes_and_compact_labels(count):
    from dashboard_support import plots
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text

    numbers = list(range(2, 2 + count))
    peaks = dict.fromkeys(numbers, np.zeros((ds.NROWS, ds.NCOLS), dtype=np.uint16))
    cells = pd.DataFrame(columns=["trial", "contact", "y", "x"])
    fig = plots.walkway(peaks, cells, numbers[-1], 500)
    FigureCanvasAgg(fig).draw()
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.images]
    assert len(panels) == count
    for ax, number in zip(panels, numbers, strict=True):
        spec = ax.get_subplotspec()
        assert ax.get_shared_x_axes().joined(panels[0], ax) or count == 1
        assert ax.get_shared_y_axes().joined(panels[0], ax) or count == 1
        assert not ax.get_title(loc="left")
        assert not ax.get_ylabel()
        label = next(text for text in ax.texts if text.get_text() == str(number))
        assert label.get_transform() == ax.transAxes
        assert label.get_ha() == "left" and label.get_va() == "top"
        x, y = label.get_position()
        assert 0 < x < 0.1 and 0.9 < y < 1
        for axis, visible in [
            (ax.xaxis, spec.is_last_row()),
            (ax.yaxis, spec.is_first_col()),
        ]:
            assert all(
                tick.tick1line.get_visible() == visible
                for tick in axis.get_major_ticks()
            )
            assert all(
                tick.label1.get_visible() == visible for tick in axis.get_major_ticks()
            )
    assert (
        sum(text.get_text() == "Across (cm)" for text in fig.findobj(match=Text)) == 1
    )
    bar = panels[-1].images[0].colorbar
    assert bar.orientation == "horizontal"
