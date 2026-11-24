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
    # PPDS: registered footprints

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    The Zenodo file `features_footprints.parquet` has the peak-pressure image of every footprint of the walking trials
    after segmentation and registration: for each subject, task and foot, every footprint was aligned
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

    The registration aligns the footprints of one subject, task and foot; footprints of different
    subjects or tasks have different templates and are not aligned to each other.
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
            SELECT footprint, trial, contact, start_frame, stop_frame
            FROM maps JOIN contacts USING (trial_id, contact)
            ORDER BY footprint
        """, [subject.value, speed.value, side.value]).df()
    _options = {
        f"Footprint {r.footprint} · Trial {r.trial} · Contact {r.contact}":
            (int(r.trial), int(r.start_frame), int(r.stop_frame))
        for r in _steps.itertuples()
    }
    mo.stop(not _options, mo.md("No registered footprints for this selection."))
    source_contact = mo.ui.dropdown(_options, value=next(iter(_options)), label="Source contact")
    mo.vstack([mo.md("Select a contact to play its original sensor readings on the trial clock."), source_contact])
    return (source_contact,)


@app.cell(hide_code=True)
def _(ds, pressure_player, side, source_contact, speed, subject):
    _trial, _start, _stop = source_contact.value
    _series = ds.pressure(subject.value, speed.value, _trial, side=side.value)
    pressure_player(
        _series[_start:_stop], ds.FREQ_HZ, first_frame=_start,
        title=f"{subject.value} · {speed.selected_key} walking · {source_contact.selected_key} · {side.selected_key} foot",
    )


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


if __name__ == "__main__":
    app.run()
