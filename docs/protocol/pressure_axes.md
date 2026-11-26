# Pressure coordinates and sampling

Pressure time series use `(frame, y, x)`. The grid has 353 rows and 64 columns. Two 176-row platforms lie end to end, and the Zebris software joins them with row 176, whose value is the mean of rows 175 and 177, rounded down to the 5 kPa step. Its readings count toward force, contact area and the pressure maps like any other row. In gait trials, `y` increases along the walkway in the direction of travel, and `x` increases toward the participant’s right when facing that direction. Use the `side` field for anatomical attribution, including crossing steps. The stance grid uses the platform sensor orientation. The center of sensor `(y=0, x=0)` is the origin. Multiply either sensor index by 8.469 to obtain its center position in millimetres.

Gait maps show the walkway from above. When travel points up the page, increasing `x` points right. When travel points right, increasing `x` points down the page.

| Task | Array shape | Sampling rate | Time interval |
|---|---|---|---|
| Gait | `(n_frames, 353, 64)` | 100 Hz | 0.01 s per frame |
| Stance | `(100, 353, 64)` | 10 Hz | Final 10 s of one recording per participant |

For a trial array `pressure` in kPa, `pressure[t]` is one sensor map, `pressure[:, y, x]` is one sensor's time series, and `pressure.max(axis=0)` gives the peak pressure at each sensor. In the sparse pressure representation, an omitted cell within a recorded frame has zero pressure. Trial lengths are needed to reconstruct the full time axis, including zero-load frames.

Registered footprint images are two-dimensional spatial measurements. An index into a collection of these images identifies a footprint. See the [data guide](../../data/README.md) for the table fields and units.
