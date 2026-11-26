# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/970500e132005f36e8fcb866a51745f026f612f7/notebook-assets/reader_data-0.2.0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/970500e132005f36e8fcb866a51745f026f612f7/notebook-assets/ppds-0.2.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: ground reaction forces of one participant at all walking speeds

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    Select a participant to plot walking trials at each speed. The force on each foot and
    the combined vertical force come from the plantar pressure in the same trial frame.
    The vertical axis can show newtons or multiples of the participant's body weight.

    ```python
    from ppds import dataset as ds
    frames = ds.pressure_features("s001", "wc", 1)
    ```
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ppds import dataset as ds
    from ppds.playback import pressure_player

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    SPEEDS = {"ws": "slow", "wc": "comfortable", "wf": "fast"}
    COLORS = {"ws": "#86b6ef", "wc": "#2a78d6", "wf": "#104281"}  # one hue, light (slow) to dark (fast)
    G = 9.81
    plt.rcParams.update({"axes.grid": True, "grid.color": "0.9", "axes.spines.top": False, "axes.spines.right": False})
    return COLORS, G, SPEEDS, ds, mo, np, plt, pressure_player


@app.cell
def _(ds, mo):
    subjects = ds.subjects().set_index("subject")
    _subjects = sorted(subjects.index)
    mo.stop(not _subjects, mo.md("No participants in this release."))
    subject = mo.ui.dropdown(options=_subjects, value=_subjects[0], label="Subject")
    body_weight = mo.ui.switch(label="Force in body weights")
    mo.hstack([subject, body_weight], justify="start")
    return body_weight, subject, subjects


@app.cell
def _(SPEEDS, ds, subject):
    walking_trials = ds.trials(subject=subject.value, condition=list(SPEEDS))
    frames = ds.pressure_features(subject.value, list(SPEEDS)).merge(
        walking_trials[["trial_id", "frequency_hz"]], on="trial_id", validate="many_to_one",
    )
    frames["time_s"] = frames.frame / frames.frequency_hz
    return frames, walking_trials


@app.cell
def _(COLORS, G, SPEEDS, body_weight, frames, plt, subject, subjects):
    _scale = 1 / (subjects.loc[subject.value, "Mass"] * G) if body_weight.value else 1.0
    _unit = "body weights" if body_weight.value else "N"
    _fig, _axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    for _row, (_code, _speed) in enumerate(SPEEDS.items()):
        _data = frames[frames.condition == _code]
        for _axis, (_label, _column) in zip(_axes[_row], (
            ("Left", "grf_z_left"), ("Right", "grf_z_right"), ("Combined", "grf_z")
        )):
            for _, _trial in _data.groupby("trial"):
                _axis.plot(_trial.time_s, _trial[_column] * _scale,
                           color=COLORS[_code], lw=1, alpha=0.8)
            _axis.set_title(f"{_label}, {_speed} ({_data.trial.nunique()} trials)", fontsize=10)
        _axes[_row, 0].set_ylabel(f"Vertical force ({_unit})")
    for _axis in _axes[-1]:
        _axis.set_xlabel("Time from trial start (s)")
    _fig.suptitle(f"{subject.value}: pressure-derived vertical ground reaction force")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Butterfly diagram

    Each contact is centred on its pressure-weighted centre, so every step of a foot lands on one
    mean footprint. The two footprints sit a fixed five sensors apart; that spacing is schematic.
    In each frame, the left and right centres of pressure are combined, weighted by the vertical force
    under each foot, which traces the transfer of load from one foot to the other.
    The first and last contacts of a pass share double support with a foot off the walkway, which the
    platform does not measure, so each pass is drawn from its first to its last measured double support.
    """)
    return


@app.cell
def _(ds, np):
    def butterfly(cells, contacts, frames):
        """Return mean centred footprints per side and one combined COP path per trial.

        cells: contact_maps rows (trial, contact, y, x, peak_pressure) of one condition.
        contacts and frames: the matching contacts and pressure_features rows.
        Coordinates are in sensor pitches; paths have NaN gaps between disjoint contacts.
        """
        pixels = cells.merge(contacts[["trial", "contact", "side"]], on=["trial", "contact"], validate="many_to_one")
        centres, feet = {}, {}
        for side, group in pixels.groupby("side"):
            positions, pressures = [], []
            for key, step in group.groupby(["trial", "contact"]):
                xy = step[["x", "y"]].to_numpy(dtype=int)
                centre = np.rint(np.average(xy, axis=0, weights=step.peak_pressure)).astype(int)
                centres[key] = centre
                positions.append(xy - centre)
                pressures.append(step.peak_pressure.to_numpy())
            xy = np.concatenate(positions)
            (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
            image = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
            np.add.at(image, (xy[:, 1] - y0, xy[:, 0] - x0), np.concatenate(pressures))
            extent = np.array([x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5])
            # Place the left foot left of the midline and the right foot right of it.
            offset = -2.5 - extent[1] if side == "L" else 2.5 - extent[0]
            feet[side] = {"image": image / len(positions), "extent": extent + [offset, offset, 0, 0], "offset": offset}

        paths = {}
        for trial, trace in frames.groupby("trial"):
            trace = trace.sort_values("frame")
            weighted = np.zeros((len(trace), 2))
            load = np.zeros(len(trace))
            support = np.zeros((len(trace), 2), dtype=int)  # contact number on each foot, 0 if unloaded
            for step in contacts[contacts.trial == trial].itertuples():
                name = {"L": "left", "R": "right"}[step.side]
                stance = trace.iloc[step.start_frame:step.stop_frame]  # stop_frame is exclusive
                cop = stance[[f"cop_x_{name}", f"cop_y_{name}"]].to_numpy() / (ds.CELL_CM * 10)  # mm to pitches
                local = cop - centres[(trial, step.contact)] + [feet[step.side]["offset"], 0]
                force = stance[f"grf_z_{name}"].to_numpy()
                loaded = force > 0
                rows = np.flatnonzero(loaded) + step.start_frame
                weighted[rows] += local[loaded] * force[loaded, None]
                load[rows] += force[loaded]
                support[rows, 0 if step.side == "L" else 1] = step.contact
            both = np.flatnonzero((support != 0).all(axis=1))
            if both.size:
                span = slice(both[0], both[-1] + 1)
                weighted, load, support = weighted[span], load[span], support[span]
            path = np.full_like(weighted, np.nan)
            np.divide(weighted, load[:, None], out=path, where=load[:, None] > 0)
            # Break the line where consecutive samples share no contact.
            shared = ((support[1:] == support[:-1]) & (support[1:] != 0)).any(axis=1)
            breaks = np.flatnonzero(~shared & (load[:-1] > 0) & (load[1:] > 0)) + 1
            paths[trial] = np.insert(path, breaks, np.nan, axis=0)
        return feet, paths

    return (butterfly,)


@app.cell
def _(SPEEDS, ds, subject):
    contacts = ds.contacts(subject.value, list(SPEEDS))
    with ds.connect() as _con:
        cells = _con.execute(
            """SELECT condition, trial, contact, y, x, peak_pressure FROM contact_maps
               WHERE subject = ? AND condition IN (SELECT unnest(?))""",
            [subject.value, list(SPEEDS)],
        ).df()
    return cells, contacts


@app.cell
def _(COLORS, SPEEDS, butterfly, cells, contacts, frames, np, plt, subject):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 5))
    for _axis, (_code, _speed) in zip(_axes, SPEEDS.items()):
        _axis.set(title=f"{_speed.capitalize()} walking", aspect="equal", xticks=[], yticks=[])
        _axis.grid(False)
        _axis.set_frame_on(False)
        _steps = contacts[contacts.condition == _code]
        if _steps.empty:
            _axis.text(0.5, 0.5, "No contacts", transform=_axis.transAxes, ha="center")
            continue
        _feet, _paths = butterfly(cells[cells.condition == _code], _steps, frames[frames.condition == _code])
        for _foot in _feet.values():
            _axis.imshow(np.ma.masked_equal(_foot["image"], 0), extent=_foot["extent"], origin="lower",
                         cmap="Greys", vmin=0, vmax=2 * _foot["image"].max(),
                         interpolation="bilinear", interpolation_stage="rgba")
        for _path in _paths.values():
            _axis.plot(*_path.T, color=COLORS[_code], lw=0.9, alpha=0.8)
        _left, _right = _feet.get("L", _feet.get("R")), _feet.get("R", _feet.get("L"))
        _axis.set_xlim(_left["extent"][0] - 2, _right["extent"][1] + 2)
    _fig.suptitle(f"{subject.value}: butterfly diagrams of the combined centre of pressure")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def choose_force_trial(SPEEDS, mo, walking_trials):
    _options = {
        f"{SPEEDS[r.condition].capitalize()} walking · Trial {r.trial}": (r.condition, int(r.trial), float(r.frequency_hz))
        for r in walking_trials.sort_values(["condition", "trial"]).itertuples()
    }
    mo.stop(not _options, mo.md("No walking trials for this participant."))
    force_trial = mo.ui.dropdown(_options, value=next(iter(_options)), label="Trial for pressure playback")
    mo.output.replace(force_trial)
    return (force_trial,)


@app.cell(hide_code=True)
def _(ds, force_trial, pressure_player, subject):
    _condition, _number, _hz = force_trial.value
    pressure_player(ds.pressure(subject.value, _condition, _number), _hz,
                    title=f"{subject.value} · {force_trial.selected_key}")


if __name__ == "__main__":
    app.run()
