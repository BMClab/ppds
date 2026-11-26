# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/5188514abd44cdda18863454b96541f932c0ab13/notebook-assets/reader_data-0.2.1-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/5188514abd44cdda18863454b96541f932c0ab13/notebook-assets/ppds-0.2.1-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: registered footprints and contact maps

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    The Zenodo file `features_footprints.parquet` has the peak-pressure image of every footprint of the walking trials
    after segmentation and registration: for each participant, task and foot, every footprint was aligned
    (centroid and principal axis) to the first footprint (`footprint` 0, the template). Values in kPa;
    only the cells with pressure are stored. Each image links to its gait contact through
    `(trial_id, contact)`. The contact row gives timing and loading on the original platform grid.

    ```python
    from ppds import dataset as ds
    f = ds.footprints("s001", "wc", "L")    # float array (footprints, 353, 64), kPa

    import pandas as pd                      # from the directory containing the Zenodo files
    d = pd.read_parquet("features_footprints.parquet",
                        filters=[("subject", "=", "s001"), ("condition", "=", "wc"), ("side", "=", "L")])
    ```

    The registration aligns the footprints of one participant, task and foot; footprints of different
    participants or tasks have different templates and are not aligned to each other.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ppds import dataset as ds
    from ppds.playback import pressure_player

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    SPEEDS = {"ws": "slow", "wc": "comfortable", "wf": "fast"}
    CELL = ds.CELL_CM
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    return CELL, SPEEDS, ds, mo, np, pd, plt, pressure_player


@app.cell
def _(SPEEDS, ds, mo):
    _subjects = ds.subjects().subject.tolist()
    mo.stop(not _subjects, mo.md("No participants in this release."))
    subject = mo.ui.dropdown(options=_subjects, value=_subjects[0], label="Subject")
    speed = mo.ui.dropdown(options={v: k for k, v in SPEEDS.items()}, value="comfortable", label="Speed")
    side = mo.ui.dropdown(options={"left": "L", "right": "R"}, value="left", label="Foot")
    mo.hstack([subject, speed, side], justify="start")
    return side, speed, subject


@app.cell
def _(ds, mo, np, plt, side, speed, subject):
    fps = ds.footprints(subject.value, speed.value, side.value)
    mo.stop(len(fps) == 0, mo.callout(mo.md(f"No footprints for {subject.value} at {speed.selected_key} speed."), kind="warn"))


    def crop(images, margin=2):
        """Rows and columns of the region with pressure in any image (for plotting)."""
        m = np.any(images > 0, axis=0)
        ys, xs = np.nonzero(m)
        return slice(max(ys.min() - margin, 0), ys.max() + margin + 1), slice(max(xs.min() - margin, 0), xs.max() + margin + 1)


    ry, rx = crop(fps)
    _n = len(fps)
    _cols = 10
    _fig, _axs = plt.subplots(int(np.ceil(_n / _cols)), _cols, figsize=(10, 2.2 * np.ceil(_n / _cols)), squeeze=False)
    for _k, _ax in enumerate(_axs.flat):
        _ax.set_xticks([])
        _ax.set_yticks([])
        if _k < _n:
            _ax.imshow(np.ma.masked_equal(fps[_k, ry, rx], 0), origin="lower", cmap="viridis", vmin=0, vmax=fps.max(), interpolation="nearest")
            _ax.contour(fps[0, ry, rx] > 0, [0.5], colors="r", linewidths=0.8)
            _ax.set_title("0 · template" if _k == 0 else str(_k), fontsize=8)
        else:
            _ax.remove()
    _fig.suptitle(f"{subject.value}, {speed.selected_key} speed, {side.selected_key} foot: {_n} footprints (red: template outline)", fontsize=10)
    _fig.tight_layout()
    _fig
    return fps, rx, ry


@app.cell(hide_code=True)
def _(ds, mo, side, speed, subject):
    with ds.connect() as _con:
        _steps = _con.execute("""
            WITH maps AS (
                SELECT DISTINCT footprint, trial_id, contact FROM footprints
                WHERE subject = ? AND condition = ? AND side = ?
            )
            SELECT m.footprint, c.trial, m.contact, c.start_frame, c.stop_frame, t.frequency_hz
            FROM maps m JOIN contacts c USING (trial_id, contact) JOIN trials t USING (trial_id)
            ORDER BY m.footprint
        """, [subject.value, speed.value, side.value]).df()
    _options = {
        f"Footprint {r.footprint} · Trial {r.trial} · Contact {r.contact}":
            (int(r.trial), int(r.start_frame), int(r.stop_frame), float(r.frequency_hz))
        for r in _steps.itertuples()
    }
    mo.stop(not _options, mo.md("No registered footprints for this selection."))
    source_contact = mo.ui.dropdown(_options, value=next(iter(_options)), label="Source contact")
    mo.vstack([mo.md("Select a contact to play its original sensor readings on the trial clock."), source_contact])
    return (source_contact,)


@app.cell(hide_code=True)
def _(ds, pressure_player, side, source_contact, speed, subject):
    _trial, _start, _stop, _hz = source_contact.value
    _series = ds.pressure(subject.value, speed.value, _trial, side=side.value)
    pressure_player(
        _series[_start:_stop], _hz, first_frame=_start,
        title=f"{subject.value} · {speed.selected_key} walking · {source_contact.selected_key} · {side.selected_key} foot",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The file `features_contact_maps.parquet` stores two maps for every walking and running contact on the
    original platform grid: the peak pressure and the pressure–time integral of each sensor over the contact,
    with unloaded frames counted as zero. The mean pressure is the integral divided by the contact time.
    `ds.contact_image(..., kind=...)` returns any of the three for the selected source contact:
    """)
    return


@app.cell
def _(ds, np, plt, side, source_contact, speed, subject):
    _trial = source_contact.value[0]
    _contact = int(source_contact.selected_key.split("Contact ")[1])
    _maps = {kind: ds.contact_image(subject.value, speed.value, _trial, _contact, kind=kind) for kind in ("peak", "pti", "mean")}
    _rows, _cols = np.nonzero(_maps["peak"])
    _ry = slice(max(_rows.min() - 2, 0), _rows.max() + 3)
    _rx = slice(max(_cols.min() - 2, 0), _cols.max() + 3)
    _fig, _axs = plt.subplots(1, 3, figsize=(8, 4.5))
    for _ax, (_kind, _label) in zip(_axs, (("peak", "Peak (kPa)"), ("pti", "Pressure–time integral (kPa·s)"), ("mean", "Mean (kPa)"))):
        _im = _ax.imshow(np.ma.masked_equal(_maps[_kind][_ry, _rx], 0), origin="lower", cmap="viridis", interpolation="nearest")
        _ax.set_xticks([])
        _ax.set_yticks([])
        _fig.colorbar(_im, ax=_ax, label=_label, shrink=0.8)
    _fig.suptitle(f"{subject.value}, {speed.selected_key} walking, {side.selected_key} foot: {source_contact.selected_key}", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mean and standard deviation across footprints (the cells without pressure in a footprint count as 0):
    """)
    return


@app.cell
def _(fps, np, plt, rx, ry):
    _fig, _axs = plt.subplots(1, 2, figsize=(6, 5))
    for _ax, _img, _lab in zip(_axs, (fps.mean(axis=0), fps.std(axis=0)), ("Mean (kPa)", "SD (kPa)")):
        _im = _ax.imshow(np.ma.masked_equal(_img[ry, rx], 0), origin="lower", cmap="viridis", interpolation="nearest")
        _ax.set_xticks([])
        _ax.set_yticks([])
        _fig.colorbar(_im, ax=_ax, label=_lab, shrink=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table summarizes the area of positive pixels and peak pressure in the registered images
    at each walking speed. For the largest area loaded simultaneously on the platform and the
    measured sensor peak during a contact, use `max_contact_area` and `peak_pressure` from `ds.contacts()`.
    """)
    return


@app.cell
def _(CELL, SPEEDS, ds, pd, subject):
    _rows = []
    for _c in SPEEDS:
        for _s in ("L", "R"):
            _f = ds.footprints(subject.value, _c, _s)
            for _k, _img in enumerate(_f):
                _rows.append({"speed": SPEEDS[_c], "foot": _s, "footprint": _k, "area_cm2": (_img > 0).sum() * CELL**2, "peak_kpa": _img.max()})
    per_footprint = pd.DataFrame(_rows)
    per_footprint.groupby(["speed", "foot"], sort=False)[["area_cm2", "peak_kpa"]].agg(["mean", "std", "count"]).round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strike index and running contacts

    Each contact row has a `strike_index`: the centre of pressure at onset (the first frame with at least
    20 N on the foot), measured from the heel edge of the contact footprint as a fraction of its length
    (after Cavanagh and Lafortune, 1980).
    Walking landings concentrate near the heel. Running landings vary between participants, from rearfoot
    to midfoot, with a few contacts further forward.
    """)
    return


@app.cell
def _(SPEEDS, ds, np, plt):
    _contacts = ds.contacts()
    _labels = {**{k: f"walking, {v}" for k, v in SPEEDS.items()}, "rc": "running"}
    _groups = [(_labels[_c], _contacts.loc[_contacts.condition == _c, "strike_index"]) for _c in _labels]
    _groups = [(_label, _values) for _label, _values in _groups if len(_values)]
    _fig, _ax = plt.subplots(figsize=(7, 3))
    for _label, _values in _groups:
        _sorted = np.sort(_values.to_numpy())
        _ax.step(_sorted, np.arange(1, len(_sorted) + 1) / len(_sorted), where="post", label=_label)
    _ax.set(xlabel="Strike index (0 heel, 1 toe)", ylabel="Fraction of contacts", xlim=(0, 1), ylim=(0, 1))
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Registered footprints are released for walking only. `ds.registered_footprints(subject, condition, side)`
    applies the same registration to the original-grid peak maps of any gait condition; for walking it
    reproduces `ds.footprints()` exactly. For running it is experimental: the alignment has not been checked
    anatomically, and the footprint shape depends on how the foot lands. The images below are the selected
    participant's running footprints for the selected foot, with the template outline in red.
    """)
    return


@app.cell
def _(ds, mo, np, plt, side, subject):
    _running = ds.trials(subject.value, "rc")
    mo.stop(_running.empty, mo.md(f"{subject.value} did not run."))
    _fps = ds.registered_footprints(subject.value, "rc", side.value)
    _mask = np.any(_fps > 0, axis=0)
    _ys, _xs = np.nonzero(_mask)
    _ry, _rx = slice(max(_ys.min() - 2, 0), _ys.max() + 3), slice(max(_xs.min() - 2, 0), _xs.max() + 3)
    _cols = 10
    _fig, _axs = plt.subplots(int(np.ceil(len(_fps) / _cols)), _cols, figsize=(10, 2.2 * np.ceil(len(_fps) / _cols)), squeeze=False)
    for _k, _ax in enumerate(_axs.flat):
        _ax.set_xticks([])
        _ax.set_yticks([])
        if _k < len(_fps):
            _ax.imshow(np.ma.masked_equal(_fps[_k, _ry, _rx], 0), origin="lower", cmap="viridis", vmin=0, vmax=_fps.max(), interpolation="nearest")
            _ax.contour(_fps[0, _ry, _rx] > 0, [0.5], colors="r", linewidths=0.8)
            _ax.set_title("0 · template" if _k == 0 else str(_k), fontsize=8)
        else:
            _ax.remove()
    _fig.suptitle(f"{subject.value}, running, {side.selected_key} foot: {len(_fps)} experimental registrations", fontsize=10)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
