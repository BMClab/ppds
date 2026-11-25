# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/4100775b8ede8154582e950908c883dfa8273087/notebook-assets/reader_data-0.1.1.dev0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/4100775b8ede8154582e950908c883dfa8273087/notebook-assets/ppds-0.1.1.dev0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Explore a PPDS trial

    Select a cohort, participant, and trial. Compare pressure-derived measurements for both
    feet together or for either foot. Choose a gait contact to locate it on the platform
    and inspect its footprint.
    """)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ppds import dataset as ds
    from ppds.playback import pressure_player

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    tasks = {
        "wc": "Comfortable walking",
        "ws": "Slow walking",
        "wf": "Fast walking",
        "rc": "Comfortable running",
        "ss": "Quiet standing",
    }
    foot_names = {"L": "Left", "R": "Right"}
    return ds, foot_names, mo, np, plt, pressure_player, tasks


@app.cell
def _(ds):
    people = ds.subjects()
    return (people,)


@app.cell(hide_code=True)
def _(mo, np, people):
    _min = int(np.floor(people.Age.min()))
    _max = int(np.ceil(people.Age.max()))
    age = mo.ui.range_slider(
        _min,
        _max,
        step=0.1,
        value=[_min, _max],
        debounce=True,
        show_value=True,
        full_width=True,
        label="Age range (years)",
    )
    characteristic = mo.ui.dropdown(
        {
            "Gender": "Gender",
            "Dominant leg": "DominantLeg",
            "Physical activity (days/week)": "PAfreq",
            "Usual footwear": "Footwear",
            "Reported illness": "Illness",
            "Surgery in past year": "Surgery",
        },
        value="Gender",
        label="Participant characteristic",
    )
    mo.vstack([age, characteristic])
    return age, characteristic


@app.cell(hide_code=True)
def response_filter(characteristic, mo, people):
    responses = (
        people[characteristic.value].astype("string").str.strip()
        .fillna("").replace("", "Missing response")
    )
    category = mo.ui.dropdown(
        ["All responses", *sorted(responses.unique())],
        value="All responses",
        label=characteristic.selected_key,
    )
    mo.output.replace(category)
    return category, responses


@app.cell(hide_code=True)
def _(age, category, mo, people, responses):
    _match = people.Age.between(*age.value)
    if category.value != "All responses":
        _match &= responses.eq(category.value)
    cohort = people.loc[_match]
    mo.stop(
        cohort.empty,
        mo.md("**0 participants match.** Adjust the age range or the characteristic."),
    )
    participant = mo.ui.dropdown(
        cohort.subject.tolist(),
        value=cohort.subject.iloc[0],
        searchable=True,
        label="Participant",
    )
    mo.vstack(
        [mo.md(f"**{len(cohort)} of {len(people)} participants** match."), participant]
    )
    return (participant,)


@app.cell(hide_code=True)
def _(ds, mo, participant, tasks):
    available = ds.trials(subject=participant.value)
    _options = {
        name: code for code, name in tasks.items() if code in set(available.condition)
    }
    condition = mo.ui.dropdown(_options, value=next(iter(_options)), label="Condition")
    mo.output.replace(condition)
    return available, condition


@app.cell(hide_code=True)
def _(available, condition, mo):
    task_trials = available.loc[available.condition == condition.value].sort_values(
        "trial"
    )
    _numbers = {str(n): int(n) for n in task_trials.trial}
    trial = mo.ui.dropdown(_numbers, value=next(iter(_numbers)), label="Trial")
    mo.output.replace(trial)
    return task_trials, trial


@app.cell(hide_code=True)
def load_recording(ds, mo, task_trials, tasks, trial):
    _row = task_trials.loc[task_trials.trial == trial.value].iloc[0]
    key = (_row.subject, _row.condition, int(_row.trial))
    hz = float(_row.frequency_hz)
    frames = ds.pressure_features(*key)
    recording = ds.pressure(*key)
    trial_peak = recording.max(axis=0)
    contacts = ds.contacts(*key)
    mo.md(
        f"## {key[0]} · {tasks[key[1]]} · Trial {key[2]}\n\n"
        f"{len(frames)} frames at {hz:g} Hz · {len(frames) / hz:.2f} s"
    )
    return contacts, frames, hz, key, recording, trial_peak


@app.cell(hide_code=True)
def _(hz, key, pressure_player, recording, tasks):
    pressure_player(recording, hz, title=f"{key[0]} · {tasks[key[1]]} · Trial {key[2]}",
                    stance=key[1] == "ss")


@app.cell(hide_code=True)
def _(contacts, foot_names, mo):
    if contacts.empty:
        contact_picker = None
        _display = mo.md(
            "Quiet standing records continuous support. The plots show the full standing trial."
        )
    else:
        _options = {
            f"{r.contact} · {foot_names[r.side]} · frames {r.start_frame}–{r.stop_frame - 1}": int(
                r.contact
            )
            for r in contacts.itertuples()
        }
        contact_picker = mo.ui.dropdown(
            _options, value=next(iter(_options)), label="Inspect contact"
        )
        _cols = {
            "contact": "Contact",
            "side": "Foot",
            "start_frame": "First frame",
            "stop_frame": "Stop (exclusive)",
            "contact_time": "Duration (s)",
            "peak_pressure": "Peak pressure (kPa)",
            "peak_grf_z": "Peak force (N)",
            "vertical_impulse": "Impulse (N·s)",
            "max_contact_area": "Max. area (cm²)",
            "contact_y_min": "Min. platform y (mm)",
            "contact_y_max": "Max. platform y (mm)",
        }
        _table = (
            contacts[list(_cols)]
            .round(
                {
                    "contact_time": 2,
                    "peak_grf_z": 2,
                    "vertical_impulse": 2,
                    "max_contact_area": 2,
                    "contact_y_min": 3,
                    "contact_y_max": 3,
                }
            )
            .rename(columns=_cols)
        )
        _display = mo.vstack(
            [
                mo.ui.table(
                    _table,
                    selection=None,
                    label="Gait contacts",
                    show_column_summaries=False,
                ),
                mo.md(
                    "The platform y bounds are the lowest and highest loaded sensor centres along the walkway."
                ),
                contact_picker,
            ]
        )
    mo.output.replace(_display)
    return (contact_picker,)


@app.cell
def _(contact_picker, contacts):
    step = (
        None
        if contact_picker is None
        else contacts.loc[contacts.contact == contact_picker.value].iloc[0]
    )
    return (step,)


@app.cell(hide_code=True)
def _(mo):
    feature = mo.ui.dropdown(
        {
            "Vertical force": [("grf_z", "Vertical force (N)")],
            "Centre of pressure": [("cop_x", "COP x (mm)"), ("cop_y", "COP y (mm)")],
            "Contact area": [("contact_area", "Contact area (cm²)")],
            "Peak pressure": [("peak_pressure", "Peak pressure (kPa)")],
        },
        value="Vertical force",
        label="Frame-level feature",
    )
    signal_styles = {
        "Combined": ("", "#30353a", "--"),
        "Left": ("_left", "#246386", "-"),
        "Right": ("_right", "#aa4c26", "-."),
    }
    signals = mo.ui.multiselect(
        list(signal_styles), value=list(signal_styles), label="Show signals"
    )
    mo.hstack([feature, signals], justify="start", wrap=True)
    return feature, signal_styles, signals


@app.cell(hide_code=True)
def _(ds, feature, frames, hz, key, mo, np, plt, signal_styles, signals, step, trial_peak):
    mo.stop(not signals.value, mo.md("Select at least one signal to show the plot."))
    _series = feature.value
    _show_cop = feature.selected_key == "Centre of pressure"
    _fig, (_mat, *_traces) = plt.subplots(
        1 + len(_series),
        1,
        figsize=(9, 2.6 + 2.3 * len(_series)),
        height_ratios=[1] + [1.4] * len(_series),
        layout="constrained",
    )
    _pitch = ds.CELL_CM
    _stance = key[1] == "ss"
    _grid = trial_peak if _stance else trial_peak.T
    _across, _up = ("x", "y") if _stance else ("y", "x")
    _axis_labels = {"x": "Across platform · x (cm)", "y": "Along platform · y (cm)"}
    _height, _width = _grid.shape
    _left, _right, _bottom, _top = 0, _width, 0, _height
    if _stance:
        # Match playback's two-sensor margin and keep absolute platform coordinates.
        _rows, _cols = np.nonzero(_grid)
        if _rows.size:
            _left, _right = max(0, _cols.min() - 2), min(_width, _cols.max() + 3)
            _bottom, _top = max(0, _rows.min() - 2), min(_height, _rows.max() + 3)
    _img = _mat.imshow(
        np.ma.masked_equal(_grid[_bottom:_top, _left:_right], 0),
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        extent=(np.array([_left, _right, _bottom, _top]) - 0.5) * _pitch,
    )
    if not _stance:
        _mat.invert_yaxis()
    _mat.set(
        xlabel=_axis_labels[_across],
        ylabel=_axis_labels[_up],
        title="Centre of pressure over the trial peak-pressure map"
        if _show_cop
        else "Peak pressure over the trial",
    )
    _fig.colorbar(_img, ax=_mat, label="kPa", shrink=0.85)
    _time = frames.frame / hz
    for _group in signals.value:
        _suffix, _color, _style = signal_styles[_group]
        _line = {"label": _group, "color": _color, "linestyle": _style}
        if _show_cop:
            # COP is stored in mm. Preserve missing samples as gaps between contacts.
            _mat.plot(
                frames[f"cop_{_across}{_suffix}"] / 10, frames[f"cop_{_up}{_suffix}"] / 10, **_line
            )
        for _ax, (_column, _) in zip(_traces, _series, strict=True):
            _ax.plot(_time, frames[_column + _suffix], **_line)
    if step is not None:
        _mat.axvspan(
            step.contact_y_min / 10 - _pitch / 2,
            step.contact_y_max / 10 + _pitch / 2,
            edgecolor="#914312",
            facecolor="none",
            linewidth=2,
        )
    for _ax, (_, _label) in zip(_traces, _series, strict=True):
        _ax.set(ylabel=_label, xlim=(0, len(frames) / hz))
        if not _show_cop:
            _ax.set_ylim(bottom=0)
        if step is not None:
            _ax.axvspan(
                step.start_frame / hz,
                step.stop_frame / hz,
                alpha=0.18,
                color="#927630",
                label=f"Contact {step.contact}",
            )
    _traces[-1].set_xlabel("Trial time (s)")
    _traces[0].legend(loc="upper right", ncols=4)
    mo.output.replace(_fig)


@app.cell
def footprint_plot(np, plt):
    def peak_panels(images):
        """Plot positive pressure on a shared color scale with the same sensor spacing."""
        fig, axes = plt.subplots(
            1,
            len(images),
            figsize=(3.4 * len(images) + 1, 4.0),
            squeeze=False,
            layout="constrained",
        )
        bounds = []
        for title, grid in images:
            rows, cols = np.nonzero(grid > 0)
            if len(rows):
                bounds.append((rows.min(), rows.max(), cols.min(), cols.max()))
            else:
                bounds.append((0, grid.shape[0] - 1, 0, grid.shape[1] - 1))
        height = max(b - a for a, b, _, _ in bounds) / 2 + 2
        width = max(d - c for _, _, c, d in bounds) / 2 + 2
        peak = max(1, max(grid.max() for _, grid in images))
        for ax, (title, grid), (a, b, c, d) in zip(axes.flat, images, bounds):
            img = ax.imshow(
                np.ma.masked_equal(grid, 0),
                origin="lower",
                interpolation="nearest",
                vmin=0,
                vmax=peak,
                cmap="viridis",
            )
            ax.set(
                title=title,
                xlabel="Grid column",
                ylabel="Grid row",
                xlim=((c + d) / 2 - width, (c + d) / 2 + width),
                ylim=((a + b) / 2 - height, (a + b) / 2 + height),
            )
        fig.colorbar(img, ax=list(axes.flat), label="Peak pressure (kPa)", shrink=0.85)
        return fig

    return (peak_panels,)


@app.cell(hide_code=True)
def _(ds, foot_names, key, mo, peak_panels, step):
    mo.stop(step is None)
    _contact = int(step.contact)
    _images = [("Platform sensors", ds.contact_image(*key, _contact))]
    if key[1] in ("ws", "wc", "wf"):
        _images.append(
            ("Registered footprint", ds.contact_image(*key, _contact, registered=True))
        )
    mo.vstack(
        [
            mo.md(
                f"### Contact {_contact} · {foot_names[step.side]} foot\n\n"
                f"Peak pressure **{step.peak_pressure:.0f} kPa** · "
                f"Peak force **{step.peak_grf_z:.1f} N** · Duration **{step.contact_time:.2f} s**"
            ),
            peak_panels(_images),
        ]
    )


@app.cell(hide_code=True)
def _(condition, mo):
    mo.stop(condition.value not in ("ws", "wc", "wf"))
    show_medians = mo.ui.checkbox(
        label="Compare median registered footprints across trials at this speed"
    )
    mo.output.replace(show_medians)
    return (show_medians,)


@app.cell(hide_code=True)
def median_footprints(condition, ds, mo, np, participant, peak_panels, show_medians):
    mo.stop(not show_medians.value)
    _images = []
    _missing = []
    for _side, _label in (("L", "Left"), ("R", "Right")):
        _maps = ds.footprints(participant.value, condition.value, _side)
        if len(_maps):
            _images.append((f"{_label} · median", np.median(_maps, axis=0)))
        else:
            _missing.append(_label.lower())
    if _images:
        _comparison = peak_panels(_images)
        if _missing:
            _comparison.suptitle(f"No registered footprints for the {_missing[0]} foot")
    else:
        _comparison = mo.md("No registered footprints for either foot at this walking speed.")
    _comparison


if __name__ == "__main__":
    app.run()
