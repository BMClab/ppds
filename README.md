# PPDS: plantar pressure data set

PPDS contains barefoot standing, walking and running plantar pressure recordings and 3D scans of both feet and lower legs from 95 adults. The BMClab at the Federal University of ABC collected the data with two Zebris FDM-1.5 platforms placed end to end and a Kinect sensor.

**[Read the example notebooks](https://bmclab.github.io/ppds/examples/)** · **[Files, units and coordinates](data/README.md)** · **[Dataset on Zenodo](https://doi.org/10.5281/zenodo.24544934)** · **[How to cite](#how-to-cite)** · **[Source on GitHub](https://github.com/BMClab/ppds)**

## Explore the dataset

Open an example to see its saved results and Python code. Choose **Edit in browser** to run it with WebAssembly, or **Open in molab** to use a cloud Python session. Browser pressure examples use all 164 trials from four participants. The participant and scan examples cover all 95. The following commands run the explorer locally against the full Zenodo release.

```bash
git clone https://github.com/BMClab/ppds.git
cd ppds
uv sync --locked
uv run marimo run examples/exploration.py
```

To reproduce the Data Descriptor’s participant summaries, tables, figures and technical validation from the complete 1.1.1 release, open the [manuscript notebook](paper/data_descriptor.py) with `uv run --locked marimo edit paper/data_descriptor.py`. The full run takes several minutes. To view the outputs used in the article without running anything, open the [saved notebook page](https://bmclab.github.io/ppds/paper/data_descriptor.html), which records when it was generated.

Use the same release in Python to read a comfortable-walking trial, its pressure features and its peak-pressure map.

```python
from ppds import dataset as ds

ds.use_zenodo(24544934)
frames = ds.pressure_features("s031", "wc", 1)
peak = ds.peak_image("s031", "wc", 1)
```

For offline analysis, download the nine files from Zenodo (479 MB total) and keep `scans.zip` intact. Set `PPDS_DATA` to the directory containing `metadata.json` or call `ds.use_local(directory)`. With pip, install both packages from the checkout using `python -m pip install ./framework .`. PPDS is not published on PyPI, and `pip install ppds` installs an unrelated project. [Python API](src/ppds/README.md)

## Coverage

Counts show participants, with trials, contacts or footprints in parentheses.

| Data | Quiet standing | Walking, slow | Walking, comfortable | Walking, fast | Running, comfortable |
|---|---:|---:|---:|---:|---:|
| Plantar pressure over time (353 × 64 sensors; trials) | 95 (95) | 95 (950) | 95 (948) | 95 (949) | 37 (370) |
| Gait contacts (left / right) | – | 95 (2,572 / 2,543) | 95 (2,189 / 2,184) | 95 (1,922 / 1,870) | 37 (649 / 646) |
| Registered peak-pressure footprints (left / right) | – | 95 (2,572 / 2,543) | 95 (2,189 / 2,184) | 95 (1,922 / 1,870) | – |
| Frame-level pressure features (trials) | 95 (95) | 95 (950) | 95 (948) | 95 (949) | 37 (370) |
| Peak-pressure and pressure–time integral maps (trials or contacts) | 95 (95) | 95 (5,115) | 95 (4,373) | 95 (3,792) | 37 (1,295) |

Gait pressure is sampled at 100 Hz. Each standing recording contributes the final 10 seconds at 10 Hz. Pressure arrays use `(frame, y, x)`, with 353 × 64 sensors per frame. Each participant also has a left and a right foot and lower-leg mesh.

The dataset is distributed under CC0 1.0 and the toolkit under the [MIT license](LICENSE).

## How to cite

Cite the dataset when you use the recordings or derived data. If you also use the Python toolkit or example notebooks, cite the software release alongside it.

- **Dataset** — Fukuchi, R., Fukuchi, C., & Duarte, M. (2026). *PPDS: plantar pressure data set* (v1.1.1). Zenodo. [10.5281/zenodo.24544934](https://doi.org/10.5281/zenodo.24544934)
- **Software** — Duarte, M., & Fukuchi, R. K. (2026). *Python toolkit and example notebooks for the plantar pressure data set (PPDS)* (v0.2.0). Zenodo. [10.5281/zenodo.24490656](https://doi.org/10.5281/zenodo.24490656)

The software DOI resolves to the latest release. The Zenodo page lists the DOI of each version.
