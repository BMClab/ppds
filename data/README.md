# PPDS release files

PPDS includes 95 participants, 3,217 gait trials and 95 quiet-standing trials. Each Parquet file below has a corresponding DuckDB view through `ppds.dataset.connect()`. Column definitions, units, keys, file sizes and SHA-256 checksums are in the [machine-readable dictionary](../src/ppds/release_1_1_1.json).

| Resource | One observation | Contents |
|---|---|---|
| `trials.parquet` | A selected trial | Participant characteristics, condition, trial number, sampling frequency, frame count and nonzero sensor count |
| `pressure.parquet` | A loaded sensor in a trial frame | Sensor coordinates, anatomical side (`L` or `R`) and plantar pressure in kPa |
| `scans.zip` | A binary STL member per foot | 190 separate foot and lower-leg meshes in metres. The reader can open a member such as `scans/s001l_foot.stl` directly from the archive. Extract the archive to obtain the individual STL files in `scans/`. |
| `features/pressure_features.parquet` | A selected trial frame | Vertical force, centre of pressure, contact area and peak pressure for both feet together and each foot individually |
| `features/contacts.parquet` | A walking or running foot contact | Side, frame interval, duration, peak pressure and force, vertical impulse, maximum contact area, longitudinal extent and strike index |
| `features/contact_maps.parquet` | A loaded sensor during a walking or running contact | Peak pressure (kPa) and pressure–time integral (kPa·s) on the original platform grid |
| `features/standing_maps.parquet` | A loaded sensor during a quiet-standing trial | Side, peak pressure (kPa) and pressure–time integral (kPa·s) over the 10 s trial |
| `features/footprints.parquet` | A loaded pixel of a registered walking footprint | Side, footprint number, source trial and contact, image coordinates and interpolated peak pressure |

When files are served from Zenodo, each feature filename has the prefix `features_`. The metadata records both its local path and its flat deposit filename.

Verify a local download with `uv run python scripts/verify_release.py --data /path/to/download`. The verifier checks file hashes, scan archive contents and consistency among the measurement tables, and it recomputes every frame feature, contact summary, pressure map and strike index from the sensor readings. It checks the links and bounds of the registered footprints but does not repeat the registration. [PPDS 1.1.1 on Zenodo](https://doi.org/10.5281/zenodo.24544934)

PPDS 1.1.1 corrects [PPDS 1.0.0](https://doi.org/10.5281/zenodo.24432612) and supersedes 1.1.0. It excludes 50 truncated gait contacts with their sensor readings: 40 that load an outermost sensor column (x = 0 or 63), which extends the curation rule that removed steps loading row 0 or 352; seven with the toes cut off at the far end of the walkway; one with the heel cut off at the start; and two recorded from the middle of stance. `metadata.json` lists them under `gait_contacts.excluded` with their 1.0.0 contact numbers. Reading counts and frame features follow from the remaining readings in the 47 trials that lost a contact. In 23 of them, later contacts move up one number, so their 1.0.0 `(trial_id, contact)` keys do not match. Footprint numbers shift in the 26 walking groups that lost a contact, and in s022 ws R and s084 wc R the excluded contact was the registration template, so every image in those two groups changed. All other footprint images are identical to 1.0.0, although some carry new contact numbers. The column rule is conservative: about half of its 40 contacts load the edge column lightly and look nearly complete. Because excluded contacts can lie between kept ones, 16 trials have two consecutive contacts of the same foot, so check `side` before pairing consecutive contacts into steps. The release adds the two pressure-map tables and the `strike_index` column of `features/contacts.parquet`. Compared with 1.1.0, the maps lack the excluded contacts and the strike index uses a 20 N onset instead of the first loaded frame. `scans.zip` is byte-identical to 1.0.0. To rebuild PPDS 1.1.1 from a 1.0.0 download, run the [build script](../scripts/build_release_1_1_1.py) from the repository root:

```sh
uv run --locked python scripts/build_release_1_1_1.py /path/to/1.0.0 /path/to/1.1.1
```

## Participants and trials

Use `subjects()` or the SQL view `subjects` for one row per participant. The physical trial table repeats the participant columns on each person's trials, which allows cohort selection directly from one Parquet file. Participant codes range from `s001` to `s095`. Trial numbers start at one within each participant and condition. The unique `trial_id` joins pressure and pressure-derived features to the trial table.

## Pressure and force

The platform grid is 353 rows by 64 columns, with 8.469 mm between sensor centres along both axes. Reconstruct an array of shape `(n_frames, 353, 64)` by initializing it to zero and filling the stored `(frame, y, x)` coordinates. The frame index starts at zero. Trial time in seconds is `frame / frequency_hz`. Gait uses 100 Hz. Each stance trial contains the final 100 frames at 10 Hz. Two 176-row platforms lie end to end, and the Zebris software fills row 176 at their junction with the mean of rows 175 and 177, rounded down to the 5 kPa step; those interpolated readings count toward force, contact area and the maps. During curation, steps that loaded the first or last row (0 or 352) were removed, so no gait reading lies on those rows. See the [coordinate conventions](../docs/protocol/pressure_axes.md).

Passing `side="L"` or `side="R"` to `ds.pressure(subject, condition, trial, side=...)` selects the corresponding sensors in the same grid and frames. The two arrays sum to the recorded pressure. All 112,277,664 loaded sensor readings have a side.

| Support | Vertical force (N) | Centre of pressure (mm) | Contact area (cm²) | Peak pressure (kPa) |
|---|---|---|---|---|
| Both feet | `grf_z` | `cop_x`, `cop_y` | `contact_area` | `peak_pressure` |
| Left foot | `grf_z_left` | `cop_x_left`, `cop_y_left` | `contact_area_left` | `peak_pressure_left` |
| Right foot | `grf_z_right` | `cop_x_right`, `cop_y_right` | `contact_area_right` | `peak_pressure_right` |

Vertical force equals the sum of cell pressures multiplied by sensor area, with units converted to newtons. Centre of pressure (COP) is the pressure-weighted mean of the sensor-center coordinates, with the center of sensor `(0, 0)` as origin. Contact area equals the number of loaded sensors multiplied by the area of one sensor. Peak pressure is the largest sensor reading in that frame. All 883,555 frames have left and right measurements. When a foot is unloaded, its force, area and peak pressure are zero and its COP is null. The same convention applies to the combined measurements during flight.

## Gait contacts and registered footprints

`ds.contacts(subject, condition, trial)` returns the contacts for a selection. The physical table has one row per `(trial_id, contact)`. Contact numbers start at one in chronological order in each trial. Of 14,575 contacts, 13,280 are from walking and 1,295 from running. Quiet standing has no discrete gait contacts. Kept contacts that reach rows 1–3 (177) or 350–351 (97) are at least 0.93 of the participant's median footprint length for that task and side and show a complete heel and toes. In quiet standing, the right heel of s020 reaches sensor row 0; that trial's mean vertical force on both feet is 1.02 times body weight, close to the cohort median of 1.04, so little load lies beyond the platform.

| Columns | Interpretation |
|---|---|
| `side`, `start_frame`, `stop_frame` | Anatomical foot (`L` or `R`) and zero-based frame interval. The start is inclusive and the stop is exclusive, as in a Python slice. |
| `contact_time` | Contact duration in seconds, equal to `(stop_frame - start_frame) / frequency_hz`, including any unloaded frames within the interval. A contact starts and ends at the platform's detection limit (5 kPa on one sensor, about 0.36 N); a 20 N force threshold, common with force plates, would shorten a median contact by 10–30 ms (2–4%) and remove about 0.05% of its impulse. The per-foot force in `pressure_features` gives thresholded times. |
| `peak_pressure`, `peak_grf_z` | Highest sensor pressure (kPa) and greatest vertical force (N) during the contact. |
| `vertical_impulse`, `max_contact_area` | Sum of the frame forces for the contact divided by the sampling frequency (N·s), and largest simultaneously loaded area (cm²). |
| `contact_y_min`, `contact_y_max` | Minimum and maximum loaded sensor-centre positions along the walkway (mm), measured on the original platform grid. |
| `strike_index` | Centre of pressure at onset, the first frame with at least 20 N on the foot, measured from the posterior edge of the contact footprint as a fraction of its length (after [Cavanagh and Lafortune, 1980](https://doi.org/10.1016/0021-9290(80)90033-0)). Zero is the heel end and one the toe end. The footprint spans the outer sensor edges of every row loaded during the contact, so a landing that never loads the heel is measured from the most posterior loaded row. The 20 N onset skips the faint readings that can precede the main landing. The first loaded frame reaches 20 N in 43% of contacts. `ppds.maps.strike_index` accepts another onset force. |

Each walking contact has one registered peak-pressure image. Use `(trial_id, contact)` to join its footprint pixels to the contact row. The `footprint` index starts at zero within each participant, walking speed and side. Index zero is that group's registration template. Image coordinates refer to the registered grid. To inspect the original sensor positions and pressure over time, select the same trial and side and slice `pressure[start_frame:stop_frame]`.

To reproduce a stored footprint from its original sensor peak map, use `ppds.contact_geometry.register_contact_peak`. The first map establishes the template. For subsequent contacts, pass that template as the second argument. The procedure centers each map by its pressure-weighted centroid, then aligns its loaded region to the template using the center of the grid as the rotation pivot. For a new analysis on arbitrary images, `ppds.registration.register_pa` uses the source centroid as its pivot.

## Pressure maps

Each map row describes one loaded sensor on the original platform grid. Sensors that are never loaded during the interval are omitted and have zero pressure. `peak_pressure` is the sensor's highest reading. `pressure_time_integral` is the sum of the sensor's readings divided by the trial's `frequency_hz`, so unloaded frames count as zero. Mean pressure is not stored. It equals the integral divided by `contacts.contact_time` for a gait contact, or by `trials.duration_s` for quiet standing.

| Table | Interval | Key |
|---|---|---|
| `contact_maps` | The contact's frames `[start_frame, stop_frame)` and only the contact's foot, for all 14,575 walking and running contacts | `(trial_id, contact, y, x)`; join `contacts` on `(trial_id, contact)` |
| `standing_maps` | All 100 frames of a quiet-standing trial, both feet, with each sensor's `side` | `(trial_id, y, x)`; join `trials` on `trial_id` |

The toolkit returns these maps as `[y, x]` arrays with `ds.contact_image(subject, condition, trial, contact, kind=...)` and `ds.standing_image(subject, kind=..., side=None)`, where `kind` is `"peak"`, `"pti"` or `"mean"`. The functions in `ppds.maps` compute the same quantities, and the strike index, from any `[frame, y, x]` series. Summed over a contact's sensors and multiplied by the sensor area, with units converted to newton-seconds, the pressure–time integral equals the contact's `vertical_impulse`.

Registered footprints are released for walking only. Running strike patterns vary between participants, from rearfoot to midfoot landings, with a few contacts further forward (see `strike_index`), and running registration has not been checked for anatomical alignment. `ds.registered_footprints(subject, condition, side)` applies the walking procedure to any gait condition. It reproduces the released walking footprints exactly and registers running contacts experimentally.

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
