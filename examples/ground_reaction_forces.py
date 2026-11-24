# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/fb5030b30fc057c180c83195b6fa71f533741148/notebook-assets/reader_data-0.1.0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/fb5030b30fc057c180c83195b6fa71f533741148/notebook-assets/ppds-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: ground reaction forces of one subject at all walking speeds

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

    from ppds import dataset as ds
    from ppds.playback import pressure_player

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    SPEEDS = {"ws": "slow", "wc": "comfortable", "wf": "fast"}
    COLORS = {"ws": "#86b6ef", "wc": "#2a78d6", "wf": "#104281"}  # one hue, light (slow) to dark (fast)
    G = 9.81
    plt.rcParams.update({"axes.grid": True, "grid.color": "0.9", "axes.spines.top": False, "axes.spines.right": False})
    return COLORS, G, SPEEDS, ds, mo, plt, pressure_player


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
def _(SPEEDS, mo, walking_trials):
    _options = {
        f"{SPEEDS[r.condition].capitalize()} walking · Trial {r.trial}": (r.condition, int(r.trial), int(r.frequency_hz))
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
