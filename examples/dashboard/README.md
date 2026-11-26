# Plantar pressure dashboard

Download [dashboard.html](dashboard.html) and open it in a modern browser with an internet connection. On startup, the page loads the Python runtime and scientific packages. Each selection then reads only its trials’ PPDS 1.1.1 measurements and foot scans from Zenodo, a few megabytes at a time.

To launch the [Python version](dashboard.py) from the repository root, run

```bash
uv run --locked marimo run examples/dashboard/dashboard.py
```

To rebuild the shareable HTML after editing the notebook or its supporting modules, run

```bash
uv run --locked python examples/dashboard/export_html.py
```

Choose **One trial** or **All trials**. Selecting a trial row highlights its walkway map, force segment, and centre-of-pressure paths. One trial lists each gait contact. All trials reports contact counts and mean ± sample SD for each foot across every recorded pass. Walking includes registered footprints and stride summaries. Running shows the original highest-peak contact for each foot in the selected trials. Quiet standing shows mean pressure cropped around both feet, with heels at the bottom and toes at the top. The gait report places left and right centre-of-pressure paths around a bilateral butterfly diagram, drawn for each walking pass from its first to its last double support on the walkway, then shows force over recorded time and the mean stance force for each foot on a gait-cycle axis. For quiet standing, the second plot row shows left, right and total vertical force over time. The third row shows the transverse and longitudinal centre-of-pressure coordinates over time. Both foot scans rotate together in one view. The dashboard uses PPDS 1.1.1 from `PPDS_DATA`, a release in `data/`, the manuscript notebook’s complete local cache, or Zenodo.
