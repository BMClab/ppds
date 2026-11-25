# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/00dea644fdc8dd42d7bc7e9a76925bed249453bc/notebook-assets/reader_data-0.1.1.dev0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/00dea644fdc8dd42d7bc7e9a76925bed249453bc/notebook-assets/ppds-0.1.1.dev0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quiet standing

    Each participant contributes the final 10 seconds of one quiet-standing trial.
    The pressure series contains 100 frames sampled at 10 Hz. Select a participant
    to see the mean pressure map and the combined and per-foot centre of pressure.
    The coordinates refer to sensor centres in the platform plane.
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

    return ds, mo, np, plt, pressure_player


@app.cell
def _(ds, mo):
    stance_trials = ds.trials(condition="ss").set_index("subject")
    mo.stop(stance_trials.empty, mo.md("No standing trials in this release."))
    subject = mo.ui.dropdown(options=stance_trials.index.tolist(), value=stance_trials.index[0], label="Participant")
    subject
    return stance_trials, subject


@app.cell
def load_stance(ds, stance_trials, subject):
    pressure = ds.standing(subject.value)
    hz = float(stance_trials.loc[subject.value, "frequency_hz"])
    return hz, pressure


@app.cell(hide_code=True)
def _(hz, pressure, pressure_player, subject):
    pressure_player(pressure, hz, title=f"{subject.value} · Quiet standing", stance=True)


@app.cell
def _(ds, hz, np, plt, pressure, subject):
    features = ds.pressure_features(subject.value, "ss", 1)
    _mean = pressure.mean(axis=0)
    _rows, _cols = np.nonzero(_mean)
    _x0, _x1 = max(0, _cols.min() - 2), min(_mean.shape[1], _cols.max() + 3)
    _y0, _y1 = max(0, _rows.min() - 2), min(_mean.shape[0], _rows.max() + 3)
    _pitch = ds.CELL_CM * 10
    _fig, (_map, _force) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    _im = _map.imshow(np.ma.masked_equal(_mean[_y0:_y1, _x0:_x1], 0), origin="lower",
                      cmap="viridis", vmin=0, interpolation="nearest",
                      extent=((_x0-.5)*_pitch, (_x1-.5)*_pitch, (_y0-.5)*_pitch, (_y1-.5)*_pitch))
    _fig.colorbar(_im, ax=_map, label="Mean pressure (kPa)")
    for _suffix, _label, _color, _style in (
        ("", "Combined", "#30353a", "--"),
        ("_left", "Left", "#246386", "-"),
        ("_right", "Right", "#aa4c26", "-."),
    ):
        _line = {"label": _label, "color": _color, "linestyle": _style}
        _map.plot(features["cop_x" + _suffix], features["cop_y" + _suffix], linewidth=1.2, **_line)
        _force.plot(features.frame / hz, features["grf_z" + _suffix], **_line)
    _map.set(xlabel="Platform x (mm)", ylabel="Platform y (mm)")
    _map.legend()
    _force.set(xlabel="Trial time (s)", ylabel="Vertical force (N)")
    _force.set_ylim(bottom=0)
    _force.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
