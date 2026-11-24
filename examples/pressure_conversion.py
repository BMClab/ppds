# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/796e7d1844209f46b787dc7a8b7442e3f8fe5961/reader_data-0.1.0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/796e7d1844209f46b787dc7a8b7442e3f8fe5961/ppds-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from importlib.resources import as_file, files
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq

    from ppds.conversion import ascii_to_parquet, npz_to_parquet, parquet_to_npz

    return Path, TemporaryDirectory, as_file, ascii_to_parquet, files, mo, np, npz_to_parquet, parquet_to_npz, plt, pq


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Convert plantar pressure recordings

    Start with a Zebris gait pressure ASCII file, follow two trials into Parquet,
    and open the results with the original PPDS NumPy layout. The bundled example
    contains synthetic pressures on the 353 × 64 sensor grid. All outputs for this
    demonstration go into a temporary directory. We assign the synthetic
    participant the example ID `s000`.

    ## 1. Read exercises from ASCII

    The importer reads the exercise intervals and sampling rate, applies the
    original notebook’s boundary cleanup, and writes two tables. Each row of
    `trials.parquet` describes a trial. Each row of `pressure.parquet` records
    a loaded sensor at one frame. The trial index retains the empty frames.
    """)
    return


@app.cell
def import_ascii(Path, TemporaryDirectory, as_file, ascii_to_parquet, files, pq):
    workspace = TemporaryDirectory(prefix="ppds-conversion-")
    work = Path(workspace.name)
    with as_file(files("ppds").joinpath("samples/demowc.txt")) as example:
        ascii_tables = ascii_to_parquet({("s000", "wc"): example}, work / "from-ascii")
    trials = pq.read_table(ascii_tables / "trials.parquet").to_pandas()
    trials
    return ascii_tables, trials, work, workspace


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Export the trials to NumPy

    Each file has one array named `data`, measured in N/cm², with time on the last
    axis. Select a trial to see its peak-pressure map and the first few readings
    in its Parquet table. The map uses a top-down view with travel to the right.
    """)
    return


@app.cell
def export_npz(Path, TemporaryDirectory, ascii_tables, mo, parquet_to_npz, work):
    archive_workspace = TemporaryDirectory(dir=work)
    archives = parquet_to_npz(ascii_tables, Path(archive_workspace.name) / "npz")
    picker = mo.ui.dropdown([p.name for p in archives], value=archives[0].name, label="Trial NPZ")
    picker
    return archive_workspace, archives, picker


@app.cell
def _(archives, ascii_tables, mo, np, picker, plt, pq, trials):
    _path = next(p for p in archives if p.name == picker.value)
    _row = trials.iloc[archives.index(_path)]
    with np.load(_path, allow_pickle=False) as _archive:
        _data = _archive["data"]
    _cells = pq.read_table(ascii_tables / "pressure.parquet", filters=[("trial_id", "=", int(_row.trial_id))])
    _peak = _data.max(axis=2)
    _y, _x = np.nonzero(_peak)
    _y0, _y1 = max(0, _y.min() - 3), min(_peak.shape[0], _y.max() + 4)
    _x0, _x1 = max(0, _x.min() - 3), min(_peak.shape[1], _x.max() + 4)
    _fig, _ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    _image = _ax.imshow(_peak[_y0:_y1, _x0:_x1].T, origin="upper",
                        extent=(_y0 - 0.5, _y1 - 0.5, _x1 - 0.5, _x0 - 0.5),
                        interpolation="nearest", cmap="viridis", vmin=0)
    _ax.set(xlabel="Along the walkway · y (sensor)", ylabel="Across · x (sensor)")
    _fig.colorbar(_image, ax=_ax, label="Peak pressure (N/cm²)")
    mo.vstack([
        mo.md(f"**{_path.name}** · `data.shape = {_data.shape}` · {len(_cells)} loaded readings"),
        _fig,
        mo.md("**First five Parquet readings** · `value` is pressure in kPa (10 kPa = 1 N/cm²)."),
        _cells.slice(0, 5).to_pandas(),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Import the NPZs and check the round trip

    The historical NPZ files contain only pressure. Here the sampling rate comes
    from the ASCII trial index. For newer frame-first NPZs with a stored
    `sampling_rate_hz`, the importer reads that rate from each file.
    """)
    return


@app.cell
def check_round_trip(Path, TemporaryDirectory, archives, ascii_tables, mo, npz_to_parquet, pq, trials, work):
    rates = {(r.subject, r.condition, r.trial): r.frequency_hz for r in trials.itertuples()}
    with TemporaryDirectory(dir=work) as _temp:
        _restored_dir = npz_to_parquet(archives, Path(_temp) / "from-npz", sampling_rate_hz=rates)
        for _name in ("trials.parquet", "pressure.parquet"):
            _original = pq.read_table(ascii_tables / _name)
            _restored = pq.read_table(_restored_dir / _name)
            if not _original.equals(_restored):
                raise ValueError(f"Round trip changed {_name}")
    mo.md(f"**Exact match** across {len(trials)} trials and {trials.n_cells.sum():,} loaded readings, "
          "including sensor coordinates, pressure values, frame counts and sampling rates.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Use your own recordings

    Give each conversion a new output directory. The following examples can
    be run independently after importing the three functions from `ppds.conversion`.

    ```python
    # Give an ASCII recording a subject and condition when its filename is arbitrary.
    ascii_to_parquet({("walker01", "wc"): "comfortable walk.txt"}, "walk-parquet")

    # Import a batch of original PPDS-style trial files acquired at 100 Hz.
    npz_to_parquet(["s001wc001.npz", "s001wc002.npz"], "my-parquet", sampling_rate_hz=100)

    # Export one trial from a local PPDS release or a converted collection.
    parquet_to_npz("walk-parquet", "my-npz", trials=[("walker01", "wc", 1)])
    ```

    For a custom NPZ filename, pass a mapping such as
    `{("walker01", "wc", 1): "walk-a.npz"}`. A collection contains the pressure
    present in its source. Exporting from the curated PPDS release reproduces
    those released readings. Historical NPZs carry no side labels or acquisition timestamps.
    """)
    return


if __name__ == "__main__":
    app.run()
