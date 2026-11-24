# PPDS: plantar pressure data set

PPDS contains barefoot standing, walking and running plantar pressure recordings and 3D scans of both feet and lower legs from 95 adults. The BMClab at the Federal University of ABC collected the data with a Zebris FDM-1.5 platform and a Kinect sensor.

**[Read the example notebooks](https://bmclab.github.io/ppds/examples/)** · **[Files, units and coordinates](data/README.md)** · **[Dataset and citation on Zenodo](https://doi.org/10.5281/zenodo.24432612)** · **[Source on GitHub](https://github.com/BMClab/ppds)**

## Explore the dataset

Open an example to see its saved results and Python code. Choose **Edit in browser** to run it with WebAssembly, or **Open in molab** to use a cloud Python session. Browser pressure examples use all 164 trials from four participants. The participant and scan examples cover all 95. The following commands run the explorer locally against the full Zenodo release.

```bash
git clone https://github.com/BMClab/ppds.git
cd ppds
uv sync --locked
uv run marimo run examples/exploration.py
```

Use the same release in Python to read a comfortable-walking trial, its pressure features and its peak-pressure map.

```python
from ppds import dataset as ds

ds.use_zenodo(24432612)
frames = ds.pressure_features("s031", "wc", 1)
peak = ds.peak_image("s031", "wc", 1)
```

For offline analysis, download the seven files from Zenodo (470 MB total) and keep `scans.zip` intact. Set `PPDS_DATA` to the directory containing `metadata.json` or call `ds.use_local(directory)`. With pip, install both packages from the checkout using `python -m pip install ./framework .`. [Python API](src/ppds/README.md)

## Coverage

Counts show participants, with trials, contacts or footprints in parentheses.

| Data | Standing still | Walking, slow | Walking, comfortable | Walking, fast | Running, comfortable |
|---|---:|---:|---:|---:|---:|
| Plantar pressure over time (353 × 64 sensors; trials) | 95 (95) | 95 (950) | 95 (948) | 95 (949) | 37 (370) |
| Gait contacts (left / right) | – | 95 (2,576 / 2,548) | 95 (2,193 / 2,193) | 95 (1,923 / 1,881) | 37 (649 / 662) |
| Registered peak-pressure footprints (left / right) | – | 95 (2,576 / 2,548) | 95 (2,193 / 2,193) | 95 (1,923 / 1,881) | – |
| Frame-level pressure features (trials) | 95 (95) | 95 (950) | 95 (948) | 95 (949) | 37 (370) |

Gait pressure is sampled at 100 Hz. Each standing recording contributes the final 10 seconds at 10 Hz. Pressure arrays use `(frame, y, x)`, with 353 × 64 sensors per frame. Each participant also has a left and a right foot and lower-leg mesh.

The dataset is distributed under CC0 1.0 and the toolkit under the [MIT license](LICENSE).
