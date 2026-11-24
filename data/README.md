# PPDS release files

PPDS includes 95 participants, 3,217 gait trials and 95 quiet-stance trials. Each Parquet file below has a corresponding DuckDB view through `ppds.dataset.connect()`. Column definitions, units, keys, file sizes and SHA-256 checksums are in the [machine-readable dictionary](../src/ppds/release_1_0_0.json).

| Resource | One observation | Contents |
|---|---|---|
| `trials.parquet` | A selected trial | Participant characteristics, condition, trial number, sampling frequency, frame count and nonzero sensor count |
| `pressure.parquet` | A loaded sensor in a trial frame | Sensor coordinates, anatomical side (`L` or `R`) and plantar pressure in kPa |
| `scans.zip` | A binary STL member per foot | 190 separate foot and lower-leg meshes in metres. The reader can open a member such as `scans/s001l_foot.stl` directly from the archive. Extract the archive to obtain the individual STL files in `scans/`. |
| `features/pressure_features.parquet` | A selected trial frame | Vertical force, center of pressure, contact area and peak pressure for both feet together and each foot individually |
| `features/contacts.parquet` | A walking or running foot contact | Side, frame interval, duration, peak pressure and force, vertical impulse, maximum contact area and longitudinal extent |
| `features/footprints.parquet` | A loaded pixel of a registered walking footprint | Side, footprint number, source trial and contact, image coordinates and interpolated peak pressure |

When files are served from Zenodo, each feature filename has the prefix `features_`. The metadata records both its local path and its flat deposit filename.

Verify a local download with `uv run python scripts/verify_release.py --data /path/to/download`. The verifier checks file hashes, scan archive contents and consistency among the measurement tables. [PPDS 1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.24432612)

## Participants and trials

Use `subjects()` or the SQL view `subjects` for one row per participant. The physical trial table repeats the participant columns on each person's trials, which allows cohort selection directly from one Parquet file. Participant codes range from `s001` to `s095`. Trial numbers start at one within each participant and condition. The unique `trial_id` joins pressure and pressure-derived features to the trial table.

## Pressure and force

The platform grid is 353 rows by 64 columns, with 8.469 mm between sensor centers along both axes. Reconstruct an array of shape `(n_frames, 353, 64)` by initializing it to zero and filling the stored `(frame, y, x)` coordinates. The frame index starts at zero. Trial time in seconds is `frame / frequency_hz`. Gait uses 100 Hz. Each stance trial contains the final 100 frames at 10 Hz. See the [coordinate conventions](../docs/protocol/pressure_axes.md).

Passing `side="L"` or `side="R"` to `ds.pressure(subject, condition, trial, side=...)` selects the corresponding sensors in the same grid and frames. The two arrays sum to the recorded pressure. All 112,590,395 loaded sensor readings have a side.

| Support | Vertical force (N) | Center of pressure (mm) | Contact area (cm²) | Peak pressure (kPa) |
|---|---|---|---|---|
| Both feet | `grf_z` | `cop_x`, `cop_y` | `contact_area` | `peak_pressure` |
| Left foot | `grf_z_left` | `cop_x_left`, `cop_y_left` | `contact_area_left` | `peak_pressure_left` |
| Right foot | `grf_z_right` | `cop_x_right`, `cop_y_right` | `contact_area_right` | `peak_pressure_right` |

Vertical force equals the sum of cell pressures multiplied by sensor area, with units converted to newtons. Center of pressure (COP) is the pressure-weighted mean of the sensor-center coordinates, with the center of sensor `(0, 0)` as origin. Contact area equals the number of loaded sensors multiplied by the area of one sensor. Peak pressure is the largest sensor reading in that frame. All 883,555 frames have left and right measurements. When a foot is unloaded, its force, area and peak pressure are zero and its COP is null. The same convention applies to the combined measurements during flight.

## Gait contacts and registered footprints

`ds.contacts(subject, condition, trial)` returns the contacts for a selection. The physical table has one row per `(trial_id, contact)`. Contact numbers start at one in chronological order in each trial. Of 14,625 contacts, 13,314 are from walking and 1,311 from running. Quiet stance has no discrete gait contacts.

| Columns | Interpretation |
|---|---|
| `side`, `start_frame`, `stop_frame` | Anatomical foot (`L` or `R`) and zero-based frame interval. The start is inclusive and the stop is exclusive, as in a Python slice. |
| `contact_time` | Contact duration in seconds, equal to `(stop_frame - start_frame) / frequency_hz`, including any unloaded frames within the interval. |
| `peak_pressure`, `peak_grf_z` | Highest sensor pressure (kPa) and greatest vertical force (N) during the contact. |
| `vertical_impulse`, `max_contact_area` | Sum of the frame forces for the contact divided by the sampling frequency (N·s), and largest simultaneously loaded area (cm²). |
| `contact_y_min`, `contact_y_max` | Minimum and maximum loaded sensor-center positions along the walkway (mm), measured on the original platform grid. |

Each walking contact has one registered peak-pressure image. Use `(trial_id, contact)` to join its footprint pixels to the contact row. The `footprint` index starts at zero within each participant, walking speed and side. Index zero is that group's registration template. Image coordinates refer to the registered grid. To inspect the original sensor positions and pressure over time, select the same trial and side and slice `pressure[start_frame:stop_frame]`.

To reproduce a stored footprint from its original sensor peak map, use `ppds.contact_geometry.register_contact_peak`. The first map establishes the template. For subsequent contacts, pass that template as the second argument. The procedure centers each map by its pressure-weighted centroid, then aligns its loaded region to the template using the center of the grid as the rotation pivot. For a new analysis on arbitrary images, `ppds.registration.register_pa` uses the source centroid as its pivot.

## Example cohort query

This query returns the highest pressure-derived left-foot force for each selected slow-walking trial among participants aged 60 or older. The frame table also includes flight and unloaded frames.

```sql
SELECT t.subject, t.trial, max(f.grf_z_left) AS peak_left_force
FROM trials AS t
JOIN pressure_features AS f USING (trial_id)
WHERE t.Age >= 60 AND t.condition = 'ws'
GROUP BY t.subject, t.trial
ORDER BY t.subject, t.trial;
```
