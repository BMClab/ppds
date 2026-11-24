# PPDS example notebooks

Start with the participant summary, trial explorer or 3D scans. Each saved page includes the code and four options:

- **Edit in browser** starts the editable notebook on GitHub Pages. **Open in marimo.app** opens a hosted WebAssembly editor. The first launch downloads Python and its packages. Use **Run all** if the cells are waiting.
- **Open in molab** opens the notebook preview. Choose **Run it now**, then **Run on server**. Sign in to molab to start a cloud session with the full release.
- **Download .py** saves the exact notebook source.

The browser participant tables and scan selector cover all 95 participants. Pressure analyses use an 8.5 MB sample with all 164 trials from s001, s005, s055 and s058. Cohort statistics in those browser examples describe these four participants. The saved pages, molab and local Python use the full release. The conversion example uses synthetic data everywhere.

To run locally, use `run` to interact with the notebook or `edit` to change its cells.

```bash
uv run marimo run examples/participant_summary.py -- --zenodo 24432612
uv run marimo run examples/exploration.py -- --zenodo 24432612
uv run marimo run examples/scans.py -- --zenodo 24432612
```

For any data-backed example, set `PPDS_DATA` to a downloaded release directory and omit the Zenodo argument. The conversion example uses its bundled synthetic recording. [Dataset on Zenodo](https://doi.org/10.5281/zenodo.24432612)

| Notebook | What it shows |
|---|---|
| [participant_summary.py](participant_summary.py) | Participant characteristics, descriptive statistics and distributions, with one observation per participant |
| [exploration.py](exploration.py) | Play a selected trial, compare combined and per-foot force, COP, area and peak pressure, and inspect gait contacts and registered walking footprints |
| [scans.py](scans.py) | Rotate both feet and lower legs for a selected participant and compare their mesh geometry |
| [subjects_and_trials.py](subjects_and_trials.py) | Subject table, selection of subjects and trials by their characteristics, data availability |
| [pressure.py](pressure.py) | Frame-by-frame gait pressure, vertical force, contact area, centre of pressure, and peak pressure across all gait trials (SQL) |
| [standing.py](standing.py) | Mean stance pressure map, bilateral force, and combined and per-foot COP trajectories |
| [footprints.py](footprints.py) | Registered footprints of one participant, mean and SD images, image area and peak pressure by walking speed |
| [ground_reaction_forces.py](ground_reaction_forces.py) | Combined and per-foot vertical force in each walking trial, in newtons or body weights |
| [queries.py](queries.py) | Cohort and trial queries, including contact times and impulses by task and side |
| [example_segmentation_registration.py](example_segmentation_registration.py) | Step-by-step tutorial of the segmentation and registration methods |
| [pressure_conversion.py](pressure_conversion.py) | Convert Zebris pressure ASCII and trial NPZ files to Parquet, then recover the original notebook layout. Runs with a bundled synthetic recording. |
