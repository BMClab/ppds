"""Figures for the local PPDS dashboard. Distances on the platform use sensor centres."""

import numpy as np
import plotly.graph_objects as go
from dashboard_support.measurements import SIDES
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ppds.dataset import CELL_CM

COLORS = {"L": "#d92323", "R": "#167c22"}
HIGHLIGHT = "#6b43a0"
PRESSURE = colormaps["turbo"].with_extremes(bad="#f3f4f4")
FOOTPRINT_SHADES = LinearSegmentedColormap.from_list(
    "footprint", ["#f2f2f2", "#bdbdbd"]
)


def walkway(peaks, cells, focus_trial, vmax):
    columns = 2 if len(peaks) > 1 else 1
    rows = (len(peaks) + columns - 1) // columns
    # Size the map area to the physical aspect of the sensor grid.
    along, across = next(iter(peaks.values())).shape
    map_height = 9.4 * across / along * rows / columns + 0.08 * (rows - 1)
    fig = Figure(figsize=(10, map_height + 1.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.025, wspace=0.015, hspace=0.015)
    maps, scale = fig.subfigures(
        2, 1, height_ratios=(map_height + 0.48, 0.57), hspace=0
    )
    axes = maps.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(peaks) :]:
        ax.set_visible(False)
    for ax, (number, grid) in zip(axes.flat, peaks.items()):
        ny, nx = grid.shape
        extent = np.array([-0.5, ny - 0.5, nx - 0.5, -0.5]) * CELL_CM
        img = ax.imshow(
            np.ma.masked_equal(grid.T, 0),
            extent=extent,
            origin="upper",
            cmap=PRESSURE,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        focused = focus_trial == number
        ax.text(
            0.012,
            0.965,
            str(number),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            weight="bold",
            color=HIGHLIGHT if focused else "#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
        )

        if focused and len(peaks) > 1:
            for spine in ax.spines.values():
                spine.set(color=HIGHLIGHT, linewidth=2)
        for contact, pixels in cells.loc[cells.trial.eq(number)].groupby("contact"):
            ymin, ymax = pixels.y.min(), pixels.y.max()
            xmin = pixels.x.min()
            ax.annotate(
                str(contact),
                ((ymin + ymax) / 2 * CELL_CM, xmin * CELL_CM),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.label_outer(remove_inner_ticks=True)
        ax.tick_params(labelsize=8)
    maps.supylabel("Across (cm)", fontsize=9)
    maps.supxlabel("Distance along the platform (cm)", fontsize=9)
    bar = scale.colorbar(
        img, cax=scale.add_axes([0.25, 0.78, 0.50, 0.16]), orientation="horizontal"
    )
    bar.ax.tick_params(labelsize=8)
    bar.set_label("Peak pressure (kPa)", fontsize=9)
    return fig


def standing_pressure(mean, vmax):
    """Show both feet in their recorded positions, with higher sensor rows toward the toes."""
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    y, x = np.nonzero(mean)
    y0, y1 = y.min(), y.max() + 1
    x0, x1 = x.min(), x.max() + 1
    crop = mean[y0:y1, x0:x1]
    pitch_mm = CELL_CM * 10
    extent = (np.array([x0, x1, y0, y1]) - 0.5) * pitch_mm
    img = ax.imshow(
        np.ma.masked_equal(crop, 0),
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=PRESSURE,
        vmin=0,
        vmax=vmax,
    )
    # Pad the view even when a loaded sensor touches the edge of the platform.
    left, right, bottom, top = extent
    margin = 2 * pitch_mm
    ax.set(
        xlim=(left - margin, right + margin),
        ylim=(bottom - margin, top + margin),
        xlabel="Across platform (mm)",
        ylabel="Along platform (mm)",
        facecolor=PRESSURE.get_bad(),
    )
    fig.colorbar(img, ax=ax, label="Mean pressure (kPa)", shrink=0.85)
    return fig


def footprints(means, counts, vmax):
    fig = Figure(figsize=(6.2, 3.5), layout="constrained")
    loaded = (
        np.nonzero(np.logical_or.reduce([grid > 0 for grid in means.values()]))
        if means
        else None
    )
    for index, (side, label) in enumerate(SIDES.items(), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        n = counts.get(side, 0)
        ax.set_title(f"{label} · {n} contact{'s' if n != 1 else ''}", fontsize=10)
        if side not in means:
            ax.text2D(0.5, 0.5, "No contacts", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue
        y, x = loaded
        y0, y1 = max(0, y.min() - 1), y.max() + 2
        x0, x1 = max(0, x.min() - 1), x.max() + 2
        grid = means[side][y0:y1, x0:x1]
        gy, gx = np.indices(grid.shape) * CELL_CM
        ax.plot_surface(
            gx,
            gy,
            np.where(grid > 0, grid, np.nan),
            cmap=PRESSURE,
            norm=Normalize(0, vmax),
            linewidth=0,
            rstride=1,
            cstride=1,
        )
        ax.set(xlim=(0, gx.max()), ylim=(0, gy.max()), zlim=(0, vmax))
        ax.set_box_aspect((gx.max(), gy.max(), gy.max() * 0.60))
        ax.view_init(elev=32, azim=-62)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="z", labelsize=7, pad=0)
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig


def contact_maps(cells, contacts, vmax):
    """Show the highest-peak original contact on each side, with identical distances and colors."""
    highest = (
        contacts.sort_values(
            ["peak_pressure", "trial", "contact"], ascending=[False, True, True]
        )
        .drop_duplicates("side")
        .set_index("side")
    )
    fig = Figure(figsize=(5.6, 3.4), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    width = height = 1
    for ax, (side, label) in zip(axes, SIDES.items()):
        if side not in highest.index:
            ax.set_title(label, fontsize=10)
            ax.text(0.5, 0.5, "No contacts", transform=ax.transAxes, ha="center")
            continue
        step = highest.loc[side]
        pixels = cells.loc[cells.trial.eq(step.trial) & cells.contact.eq(step.contact)]
        x = pixels.x.to_numpy(dtype=int) - int(pixels.x.min())
        y = pixels.y.to_numpy(dtype=int) - int(pixels.y.min())
        grid = np.zeros((y.max() + 1, x.max() + 1))
        grid[y, x] = pixels.peak_pressure
        ny, nx = grid.shape
        width, height = max(width, nx), max(height, ny)
        ax.imshow(
            np.ma.masked_equal(grid, 0),
            origin="lower",
            cmap=PRESSURE,
            vmin=0,
            vmax=vmax,
            extent=np.array([-nx / 2, nx / 2, -0.5, ny - 0.5]) * CELL_CM,
            interpolation="nearest",
        )
        ax.set_title(
            f"{label} · trial {int(step.trial)}, contact {int(step.contact)}",
            fontsize=9,
        )
    axes[0].set(
        xlim=(-width * CELL_CM / 2, width * CELL_CM / 2),
        ylim=(-0.5 * CELL_CM, (height - 0.5) * CELL_CM),
        ylabel="Along contact (cm)",
    )
    fig.supxlabel("Across contact (cm)", fontsize=10)
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap=PRESSURE),
        ax=axes,
        label="Peak pressure (kPa)",
        shrink=0.82,
    )
    return fig


def _cop_panels(axes, diagram, focus_trial):
    """Show identical grayscale footprints and spatial scales in the three COP panels."""
    left, middle, right = axes
    side_axes = {"L": left, "R": right}
    middle.set_title("Butterfly diagram")
    for side, ax in side_axes.items():
        ax.set_title(f"{SIDES[side]} foot")
    for ax in axes:
        ax.set(aspect="equal", xticks=[], yticks=[])
    if not diagram.feet:
        middle.text(
            0.5, 0.5, "No recorded contacts", transform=middle.transAxes, ha="center"
        )
        return

    maximum = max(foot.image.max() for foot in diagram.feet.values())
    extents = []
    paired_extents = []
    for side, foot in diagram.feet.items():
        paired = foot.extent + [foot.offset, foot.offset, 0, 0]
        for ax, extent in [(side_axes[side], foot.extent), (middle, paired)]:
            ax.imshow(
                np.ma.masked_equal(foot.image, 0),
                extent=extent,
                origin="lower",
                cmap=FOOTPRINT_SHADES,
                vmin=0,
                vmax=maximum,
                interpolation="bilinear",
                interpolation_stage="rgba",
            )
        extents.append(foot.extent)
        paired_extents.append(paired)

    for side, ax in side_axes.items():
        if side not in diagram.feet:
            ax.text(0.5, 0.5, "No contacts", transform=ax.transAxes, ha="center")
    for key, xy in diagram.paths.items():
        trial, _ = key
        side = diagram.sides[key]
        focused = trial == focus_trial
        side_axes[side].plot(
            *xy.T,
            color=COLORS[side],
            lw=1.1 if focused else 0.75,
            alpha=1 if focused else 0.7,
            zorder=3 if focused else 2,
        )
    for trial, xy in diagram.butterfly.items():
        focused = trial == focus_trial
        middle.plot(
            *xy.T,
            color="#1949c7",
            lw=1.1 if focused else 0.75,
            alpha=1 if focused else 0.7,
            zorder=3 if focused else 2,
        )

    bottom = min(e[2] for e in extents) - 2
    top = max(e[3] for e in extents) + 2
    for ax, bounds, ratio in [
        (left, extents, 0.70),
        (middle, paired_extents, 1.20),
        (right, extents, 0.70),
    ]:
        xmin, xmax = min(e[0] for e in bounds) - 1, max(e[1] for e in bounds) + 1
        centre = (xmin + xmax) / 2
        half = max(xmax - xmin, (top - bottom) * ratio) / 2
        ax.set(xlim=(centre - half, centre + half), ylim=(bottom, top))


def _gait_force_trace(ax, trials, frames, focus_trial):
    offset = 0.0
    multiple = len(trials) > 1
    for index, trial in enumerate(trials.itertuples()):
        trace = frames.loc[frames.trial_id.eq(trial.trial_id)].sort_values("frame")
        time = offset + trace.frame.to_numpy() / trial.frequency_hz
        for side, name in SIDES.items():
            ax.plot(
                time,
                trace[f"grf_z_{name.lower()}"],
                color=COLORS[side],
                lw=1.15,
                label=name if index == 0 else None,
            )
        if multiple:
            if index:
                ax.axvline(offset, lw=0.65, color="#888", zorder=0)
            if focus_trial == trial.trial:
                ax.axvspan(
                    offset, offset + trial.duration_s, color=HIGHLIGHT, alpha=0.08
                )
        offset += trial.duration_s
    ax.set(
        title="Vertical ground reaction force",
        xlim=(0, offset),
        ylim=(0, None),
        ylabel="Force (N)",
        xlabel="Cumulative recorded time (s)" if multiple else "Time (s)",
    )
    ax.legend(loc="upper right", ncols=2, fontsize=8)


def _stance_force_panels(axes, curves):
    for ax, side in zip(axes, ("R", "L")):
        items = curves[side]
        n = len(items)
        ax.set(
            title=f"{SIDES[side]} vertical force · {n} stride{'s' if n != 1 else ''}",
            xlim=(0, 100),
            xlabel="Gait cycle (%)",
            ylabel="Force (N)",
        )
        if not n:
            ax.text(0.5, 0.5, "No complete stride", transform=ax.transAxes, ha="center")
            continue
        values = np.stack([force for force, _ in items])
        phase = np.linspace(
            0, np.mean([toe_off for _, toe_off in items]), values.shape[1]
        )
        mean = values.mean(axis=0)
        if n > 1:
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(phase, mean - sd, mean + sd, color="#ccc", lw=0)
        ax.plot(phase, mean, color=COLORS[side], lw=1.7)
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))


def gait_report(trials, frames, diagram, curves, focus_trial):
    fig = Figure(figsize=(10.2, 7.8), layout="constrained")
    rows = fig.add_gridspec(3, 1, height_ratios=(2.1, 1, 1.05), hspace=0.09)
    top = rows[0].subgridspec(1, 3, width_ratios=(1, 1.7, 1))
    _cop_panels([fig.add_subplot(top[i]) for i in range(3)], diagram, focus_trial)
    _gait_force_trace(fig.add_subplot(rows[1]), trials, frames, focus_trial)
    bottom = rows[2].subgridspec(1, 2)
    right = fig.add_subplot(bottom[0])
    left = fig.add_subplot(bottom[1], sharey=right)
    _stance_force_panels((right, left), curves)
    return fig


def standing_time_series(trial, frames):
    """Keep bilateral force and the two spatial COP coordinates on the same recorded time axis."""
    trace = frames.sort_values("frame")
    time = trace.frame.to_numpy() / trial.frequency_hz
    force = Figure(figsize=(10.2, 2.6), layout="constrained")
    force_ax = force.subplots()
    cop = Figure(figsize=(10.2, 2.9), layout="constrained")
    across, along = cop.subplots(1, 2, sharex=True)
    for ax, column in [(force_ax, "grf_z"), (across, "cop_x"), (along, "cop_y")]:
        for name, suffix, color, style in [
            ("Left", "_left", COLORS["L"], "-"),
            ("Right", "_right", COLORS["R"], "-"),
            ("Both feet", "", "#1949c7", "--"),
        ]:
            ax.plot(
                time, trace[column + suffix], color=color, ls=style, lw=1.3, label=name
            )
        ax.set(xlim=(0, trial.duration_s), xlabel="Time (s)")
    force_ax.set(ylabel="Vertical force (N)", ylim=(0, None))
    across.set(title="Across platform (x)", ylabel="COP position (mm)")
    along.set(title="Along platform (y)", ylabel="COP position (mm)")
    for figure, ax in [(force, force_ax), (cop, across)]:
        figure.legend(
            *ax.get_legend_handles_labels(),
            loc="outside upper center",
            ncols=3,
            frameon=False,
            fontsize=9,
        )
    return force, cop


def _orient_scan(vertices, faces):
    cm = vertices * 100
    # Resolve the sign of the long horizontal axis using the forefoot and upper leg.
    centres = cm[faces].mean(axis=1)
    _, vecs = np.linalg.eigh(np.cov(cm[:, :2].T))
    along = vecs[:, -1]
    position = centres[:, :2] @ along
    z = centres[:, 2]
    if (
        position[z < np.quantile(z, 0.12)].mean()
        < position[z > np.quantile(z, 0.8)].mean()
    ):
        along = -along
    # A rotation preserves handedness. Positive y runs from heel toward toes.
    x = cm[:, :2] @ [along[1], -along[0]]
    y = cm[:, :2] @ along
    return x, y - y.min(), cm[:, 2] - cm[:, 2].min()


def plantar_scan(meshes):
    fig = go.Figure()
    labels = []
    gap_cm = 5.0
    for side, label in SIDES.items():
        vertices, faces = meshes[side]
        x, y, z = _orient_scan(vertices, faces)
        x = x - x.max() - gap_cm / 2 if side == "L" else x - x.min() + gap_cm / 2
        i, j, k = faces.T
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                name=label,
                color="#a8bfc2",
                lighting={
                    "ambient": 0.38,
                    "diffuse": 0.78,
                    "specular": 0.16,
                    "roughness": 0.8,
                },
                lightposition={"x": -600, "y": -800, "z": -1500},
                hoverinfo="name",
                showlegend=False,
            )
        )
        labels.append(
            {
                "x": (x.min() + x.max()) / 2,
                "y": 0,
                "z": 0,
                "text": label,
                "showarrow": False,
                "yshift": 26,
            }
        )
    clear = "rgba(0,0,0,0)"
    axis = {"visible": False, "showbackground": False}
    fig.update_layout(
        height=310,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        paper_bgcolor=clear,
        plot_bgcolor=clear,
        scene={
            "bgcolor": clear,
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": axis,
            "aspectmode": "data",
            "annotations": labels,
            "camera": {
                "eye": {"x": 0.12, "y": -0.52, "z": -1.45},
                "up": {"x": 0, "y": -1, "z": 0},
                "projection": {"type": "orthographic"},
            },
        },
    )
    return fig
