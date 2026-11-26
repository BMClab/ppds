# PPDS example notebooks

Start with the participant summary, trial explorer or 3D scans. Each saved page includes the code and four options:

- **Edit in browser** starts the editable notebook on GitHub Pages. **Open in marimo.app** opens a hosted WebAssembly editor. The first launch downloads Python and its packages. Use **Run all** if the cells are waiting.
- **Open in molab** opens the notebook preview. Choose **Run it now**, then **Run on server**. Sign in to molab to start a cloud session with the full release.
- **Download .py** saves the exact notebook source.

The browser participant tables and scan selector cover all 95 participants. Pressure analyses use a 9.0 MB sample with all 164 trials from s001, s005, s055 and s058. Cohort statistics in those browser examples describe these four participants. The saved pages, molab and local Python use the full release. The conversion example uses synthetic data everywhere.

To run locally, use `run` to interact with the notebook or `edit` to change its cells.

```bash
uv run marimo run examples/participant_summary.py
uv run marimo run examples/exploration.py
uv run marimo run examples/scans.py
```

Local runs use the release in `PPDS_DATA` or the checkout’s `data/` directory when available. Otherwise they read Zenodo record `24544934` automatically. To select that record even with a local release configured, append `-- --zenodo 24544934`. [Dataset on Zenodo](https://doi.org/10.5281/zenodo.24544934)

| Notebook | What it shows |
|---|---|
| [participant_summary.py](participant_summary.py) | Participant characteristics, descriptive statistics and distributions, with one observation per participant |
| [exploration.py](exploration.py) | Play a selected trial, compare combined and per-foot force, COP, area and peak pressure, and inspect gait contacts and registered walking footprints |
| [scans.py](scans.py) | Rotate both feet and lower legs for a selected participant and compare their mesh geometry |
| [subjects_and_trials.py](subjects_and_trials.py) | Participant table, selection of participants and trials by their characteristics, data availability |
| [pressure.py](pressure.py) | Frame-by-frame gait pressure, vertical force, contact area, centre of pressure, and peak pressure across all gait trials (SQL) |
| [standing.py](standing.py) | Mean quiet-standing pressure map, bilateral force, and combined and per-foot COP trajectories |
| [footprints.py](footprints.py) | Registered footprints of one participant, mean and SD images, image area and peak pressure by walking speed, peak, pressure–time integral and mean contact maps, strike index by task, and experimental running registration |
| [ground_reaction_forces.py](ground_reaction_forces.py) | Combined and per-foot vertical force in each walking trial, in newtons or body weights |
| [queries.py](queries.py) | Cohort and trial queries, including contact times and impulses by task and side |
| [example_segmentation_registration.py](example_segmentation_registration.py) | Step-by-step tutorial of the segmentation and registration methods |
| [pressure_conversion.py](pressure_conversion.py) | Convert Zebris pressure ASCII and trial NPZ files to Parquet, then recover the original notebook layout. Runs with a bundled synthetic recording. |
