# PPDS toolkit

The [dataset module](dataset.py) provides participant and trial selection, pressure arrays, frame-level features, gait contacts, pressure maps, registered walking footprints and meshes. Open the default local release with `from ppds import dataset as ds`. Select a download with `ds.use_local(path)`, a published Zenodo version with `ds.use_zenodo(record_id)`, or an HTTP mirror with `ds.use_url(base_url)`. Pass `flat=True` for a mirror serving the flat deposit filenames.

`ds.use_local()` reads `PPDS_DATA` each time you call it. When that variable is unset or blank, the reader uses the checkout’s `data/` directory or, in a wheel installation, the current working directory’s `data/` directory.

- `ds.subjects()` returns one row per participant and checks the repeated descriptors for consistency. `ds.trials()` returns one row per trial.
- `ds.pressure(subject, condition, trial)` reconstructs a frame-first grid in kPa. Pass `side="L"` or `side="R"` to select one foot on the same platform grid. `ds.peak_image(subject, condition, trial)` retrieves the maximum pressure at each sensor across the trial. `ds.standing(subject)` returns the 100-frame stance series.
- `ds.pressure_features(subject, condition, trial)` returns the combined and per-foot vertical force, COP, contact area and peak pressure on the trial clock. A scalar or list filter with no matches returns an empty feature table.
- `ds.contacts(subject, condition, trial)` returns walking and running contact timing, loading, longitudinal extent and strike index. Each argument is optional. The SQL view `contacts` joins to `trials` on `trial_id`.
- `ds.contact_image(subject, condition, trial, contact)` returns the peak pressure at each platform sensor during one contact, using only that foot’s readings. Pass `kind="pti"` for the pressure–time integral in kPa·s or `kind="mean"` for the mean pressure in kPa over the contact time. For a walking contact, pass `registered=True` to retrieve its aligned peak-pressure footprint in kPa. Contact numbers start at one.
- `ds.standing_image(subject)` returns the quiet-standing peak-pressure map over the 10 s trial. It accepts the same `kind` values and an optional `side="L"` or `side="R"`.
- `ds.footprints(subject, condition, side)` returns registered walking images in footprint-index order for a single participant, condition and foot. Foot-side letters accept either case. Use the SQL view `footprints` for selections spanning multiple groups and join to `contacts` on `(trial_id, contact)` to identify each image. `ds.registered_footprints(subject, condition, side)` computes the same registration from the sensor readings for any gait condition. It reproduces the released walking footprints and registers running contacts experimentally, without a check of anatomical alignment.
- The [maps module](maps.py) computes peak-pressure, pressure–time integral and mean-pressure maps and the strike index from any `[frame, y, x]` series.
- `ds.scan(subject, side)` returns vertices in metres and triangular faces. Individual STLs and the distributed scan archive preserve the same meshes.
- `ds.connect()` opens DuckDB with PPDS views. The [generic reader](../../framework/src/reader_data/README.md) exposes a release's physical tables and works independently of PPDS.

In a marimo cell, use `pressure_player` to replay a trial or a contact on the trial clock. After the cell loads, seeking and playback run in the browser. Pass `stance=True` for a standing recording or `first_frame=start` when displaying a slice beginning at frame `start`.

```python
from ppds.playback import pressure_player

pressure_player(ds.pressure("s001", "wc", 1), 100, title="s001 · Comfortable walking · Trial 1")
```

See the [file guide](../../data/README.md) for units, keys and sampling conventions, or the [marimo examples](../../examples/README.md) for worked analyses.

## Convert pressure recordings

The [conversion functions](conversion.py) accept Zebris gait pressure ASCII exports and prepared NPZ trials on the PPDS 353 × 64 grid. Each output goes into a new directory. Imports write `trials.parquet`, sparse integer-kPa `pressure.parquet`, and a small reader manifest. Open the result with `ds.use_local(directory)` and use `ds.trials()`, `ds.pressure()`, or `ds.connect()`.

```python
from ppds.conversion import ascii_to_parquet, npz_to_parquet, parquet_to_npz

ascii_to_parquet({("walker01", "wc"): "walk.txt"}, "ascii-tables")
npz_to_parquet(["s001wc001.npz", "s001wc002.npz"], "npz-tables", sampling_rate_hz=100)
parquet_to_npz("ascii-tables", "trial-files", trials=[("walker01", "wc", 1)])
```

The importers infer trial identity from PPDS filenames such as `s001wc.txt` and `s001wc001.npz`. Use an explicit mapping for custom filenames, keyed by `(subject, condition)` for ASCII or `(subject, condition, trial)` for NPZ.

ASCII import rounds each exercise’s start and end to the nearest acquisition frame (an exact half-frame time goes to the even frame), reads the half-open interval in the order listed in the export, and applies the original gait notebook’s boundary cleanup. In `metadata.json`, `ascii_exercises` records each exercise’s acquisition start frame, original frame count, first retained frame and any boundary warnings. The trial table gives the length of the cleaned array. An exercise that the cleanup leaves without loaded frames is still a trial, with zero frames and a `first_retained_frame` of null. `read_zebris_text` saves one NPZ per exercise and refuses to overwrite existing archives unless `overwrite=True`. NPZ import preserves every saved frame, including empty frames. It accepts frames first or last, checks any declared axis order, and requires pressure values on the 0.1 N/cm² grid. The sampling rate comes from the archive when available. Otherwise pass a rate or a dictionary keyed by `(subject, condition, trial)`.

Export accepts a release or a directory containing just `pressure.parquet` and `trials.parquet`. It writes the historical single `data` array in N/cm² with shape `(353, 64, frames)`. Keep the trial index for sampling rates. The export carries the pressure values present in the source tables. See the [runnable conversion notebook](../../examples/pressure_conversion.py).
