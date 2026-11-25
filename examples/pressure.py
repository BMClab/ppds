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
    mo.md(r"""
    # PPDS: plantar pressure over time

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    The pressure of every frame of every trial is in `data/pressure.parquet`
    (one row per loaded sensor and frame, in kPa). This notebook loads one
    trial as a 3D array, plays it, and computes basic variables: vertical force, contact area, centre
    of pressure (COP) and the peak-pressure image. It then compares peak pressure across
    all walking and running trials.

    ```python
    from ppds import dataset as ds
    p = ds.pressure("s001", "wc", 1)        # uint16 array (frames, 353, 64), kPa; time (s) = frame / 100
    peak = ds.peak_image("s001", "wc", 1)   # (353, 64), kPa

    features = ds.pressure_features("s001", "wc", 1)  # force, COP, contact area and peak pressure
    left = ds.pressure("s001", "wc", 1, side="L")
    contacts = ds.contacts("s001", "wc", 1)
    step = contacts.loc[contacts.side == "L"].iloc[0]
    left_step = left[step.start_frame:step.stop_frame]
    ```

    Axes: `y` (0–352) runs along the walkway and increases in the walking direction; `x` (0–63) runs
    across it; each sensor cell is 0.8469 × 0.8469 cm.
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

    CONDITIONS = {"ws": "walking, slow", "wc": "walking, comfortable", "wf": "walking, fast", "rc": "running, comfortable"}
    CELL = ds.CELL_CM  # cm
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    return CELL, CONDITIONS, ds, mo, np, plt, pressure_player


@app.cell
def _(CONDITIONS, ds, mo):
    gait_trials = ds.trials(condition=list(CONDITIONS))
    mo.stop(gait_trials.empty, mo.md("No walking or running trials are available in this release."))
    _subjects = sorted(gait_trials.subject.unique())
    subject = mo.ui.dropdown(options=_subjects, value=_subjects[0], label="Subject")
    subject
    return gait_trials, subject


@app.cell
def _(CONDITIONS, gait_trials, mo, subject):
    _available = set(gait_trials.loc[gait_trials.subject == subject.value, "condition"])
    _options = {name: code for code, name in CONDITIONS.items() if code in _available}
    _default = CONDITIONS["wc"] if "wc" in _available else next(iter(_options))
    condition = mo.ui.dropdown(options=_options, value=_default, label="Task")
    condition
    return (condition,)


@app.cell
def _(condition, ds, mo, subject):
    _trials = ds.trials(subject=subject.value, condition=condition.value)
    _opts = [str(k) for k in _trials.trial]
    frequency = _trials.set_index("trial").frequency_hz
    trial = mo.ui.dropdown(options=_opts, value=_opts[0] if _opts else None, label="Trial")
    trial
    return frequency, trial


@app.cell
def trial_signals(CELL, CONDITIONS, condition, ds, frequency, mo, np, subject, trial):
    mo.stop(trial.value is None, mo.md("_Choose a trial._"))
    p = ds.pressure(subject.value, condition.value, int(trial.value))
    hz = float(frequency.loc[int(trial.value)])
    t = np.arange(p.shape[0]) / hz
    force = p.sum(axis=(1, 2)) * 0.1 * CELL**2           # N: kPa x 0.1 (N/cm^2 per kPa) x cell area (cm^2)
    area = (p > 0).sum(axis=(1, 2)) * CELL**2            # cm^2
    _w = p.sum(axis=(1, 2)).astype(float)
    _y, _x = np.mgrid[0:p.shape[1], 0:p.shape[2]]
    with np.errstate(invalid="ignore"):
        cop_x = (p * _x).sum(axis=(1, 2)) / _w * CELL    # cm
        cop_y = (p * _y).sum(axis=(1, 2)) / _w * CELL
    mo.md(f"`{subject.value}` {CONDITIONS[condition.value]} trial {trial.value}: array {p.shape} ({p.dtype}), "
          f"{len(p) / hz:.2f} s, peak pressure {p.max()} kPa, peak force {force.max():.0f} N")
    return area, cop_x, cop_y, force, hz, p, t


@app.cell(hide_code=True)
def _(CONDITIONS, condition, hz, p, pressure_player, subject, trial):
    pressure_player(p, hz, title=f"{subject.value} · {CONDITIONS[condition.value]} · Trial {trial.value}")


@app.cell
def _(force, mo, p):
    frame = mo.ui.slider(0, p.shape[0] - 1, value=int(force.argmax()), step=1, label="Frame for detailed plots", full_width=True, show_value=True)
    frame
    return (frame,)


@app.cell
def _(CELL, area, cop_x, cop_y, force, frame, np, p, plt, t):
    _rows = np.flatnonzero(p.max(axis=(0, 2)) > 0)  # walkway rows with pressure in this trial
    _y0, _y1 = max(_rows.min() - 3, 0), min(_rows.max() + 4, p.shape[1])
    _fig = plt.figure(figsize=(10, 6.5))
    _gs = _fig.add_gridspec(3, 1, height_ratios=[1.1, 1, 1])
    _ax0 = _fig.add_subplot(_gs[0])
    _im = _ax0.imshow(p[frame.value, _y0:_y1].T, origin="lower", cmap="viridis", vmin=0, vmax=p.max(), interpolation="nearest",
                      extent=(_y0 - 0.5, _y1 - 0.5, -0.5, p.shape[2] - 0.5))
    _ax0.plot(cop_y[frame.value] / CELL, cop_x[frame.value] / CELL, "r+", ms=12, mew=2)
    _ax0.invert_yaxis()  # In a top-down view with travel to the right, x increases downward.
    _ax0.set_title(f"t = {t[frame.value]:.2f} s (direction of travel →)", fontsize=10)
    _ax0.set_xlabel("y (sensor)")
    _ax0.set_ylabel("x (sensor)")
    _fig.colorbar(_im, ax=_ax0, label="Pressure (kPa)", fraction=0.03)
    for _k, (_v, _lab) in enumerate(((force, "Vertical force (N)"), (area, "Contact area (cm²)")), start=1):
        _a = _fig.add_subplot(_gs[_k])
        _a.plot(t, _v, color="#2a78d6", lw=1.5)
        _a.axvline(t[frame.value], color="0.5", lw=1)
        _a.set_ylabel(_lab)
        _a.grid(color="0.9")
    _a.set_xlabel("Time (s)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Peak-pressure image and centre of pressure

    The peak-pressure image is the maximum of each sensor over all frames; the COP trajectory is drawn on top.
    """)
    return


@app.cell
def _(CELL, condition, cop_x, cop_y, ds, np, plt, subject, trial):
    peak = ds.peak_image(subject.value, condition.value, int(trial.value))
    _fig, _ax = plt.subplots(figsize=(4, 8))
    _im = _ax.imshow(np.ma.masked_equal(peak, 0), origin="lower", cmap="viridis", interpolation="nearest")
    _ax.plot(cop_x / CELL, cop_y / CELL, color="r", lw=1)
    _ax.set_xlabel("x (sensor)")
    _ax.set_ylabel("y (sensor, direction of travel ↑)")
    _fig.colorbar(_im, ax=_ax, label="Peak pressure (kPa)", shrink=0.6)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Peak-pressure images of all trials of this participant and task:
    """)
    return


@app.cell
def _(condition, ds, np, plt, subject):
    _tr = ds.trials(subject=subject.value, condition=condition.value)
    _peaks = [ds.peak_image(subject.value, condition.value, k) for k in _tr.trial]
    _fig, _axs = plt.subplots(1, len(_peaks), figsize=(1 + 0.9 * len(_peaks), 5), sharey=True, squeeze=False)
    _vmax = max(q.max() for q in _peaks)
    for _ax, _k, _q in zip(_axs[0], _tr.trial, _peaks):
        _im = _ax.imshow(np.ma.masked_equal(_q, 0), origin="lower", cmap="viridis", vmin=0, vmax=_vmax, interpolation="nearest")
        _ax.set_title(str(_k), fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.colorbar(_im, ax=_axs[0].tolist(), label="Peak pressure (kPa)", shrink=0.8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Peak pressure by gait condition

    Each value is the highest pressure recorded in one walking or running trial.
    ```sql
    SELECT subject, condition, trial, max(peak_pressure) AS peak_kpa
    FROM pressure_features WHERE condition IN ('ws', 'wc', 'wf', 'rc') GROUP BY ALL
    ```
    """)
    return


@app.cell
def _(CONDITIONS, ds, mo, plt):
    with ds.connect() as con:
        peaks = con.sql("""
            SELECT subject, condition, trial, max(peak_pressure) AS peak_kpa
            FROM pressure_features WHERE condition IN ('ws', 'wc', 'wf', 'rc') GROUP BY ALL
        """).df()
    _order = [c for c in CONDITIONS if c in set(peaks.condition)]
    mo.stop(not _order, mo.md("No walking or running peak-pressure measurements are available."))
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.boxplot([peaks.loc[peaks.condition == c, "peak_kpa"] for c in _order], tick_labels=[CONDITIONS[c] for c in _order],
                medianprops={"color": "0.2"}, patch_artist=True, boxprops={"facecolor": "0.85", "color": "0.4"})
    _ax.set_ylabel("Peak pressure of the trial (kPa)")
    _ax.grid(axis="y", color="0.9")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
