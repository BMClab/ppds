# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/09f021228ae66ea5dfecded8939a4dab7cde7801/notebook-assets/reader_data-0.1.1-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/09f021228ae66ea5dfecded8939a4dab7cde7801/notebook-assets/ppds-0.1.1-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: questions about the whole dataset with SQL

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    The Parquet files can be queried with SQL without loading them into memory, e.g. with
    [DuckDB](https://duckdb.org) (also available in R and other languages). `ds.connect()` returns a
    DuckDB connection with one view per table (`subjects`, `trials`, `pressure`, `pressure_features`,
    `contacts`, `footprints`, `standing`). You can also query the physical files directly:

    ```python
    import duckdb
    duckdb.sql("SELECT condition, count(*) FROM 'data/trials.parquet' GROUP BY condition").df()
    duckdb.sql("SELECT max(value) FROM read_parquet('data/pressure.parquet')").df()
    ```

    Results are read with `.df()` (pandas) or `.fetchall()`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from ppds import dataset as ds

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    con = ds.connect()
    SPEED = {"ws": "slow", "wc": "comfortable", "wf": "fast", "rc": "running"}
    COLORS = {"ws": "#86b6ef", "wc": "#2a78d6", "wf": "#104281"}
    plt.rcParams.update({"axes.grid": True, "grid.color": "0.9", "axes.spines.top": False, "axes.spines.right": False})
    return COLORS, SPEED, con, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. How many participants and trials per task?**
    """)
    return


@app.cell
def _(con):
    con.sql("""
        SELECT condition, count(DISTINCT subject) AS participants, count(*) AS trials, round(avg(duration_s), 2) AS mean_duration_s
        FROM trials GROUP BY condition ORDER BY condition
    """).df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Number and mean duration of selected gait trials by age group, gender and task.**
    """)
    return


@app.cell
def _(con):
    con.sql("""
        SELECT AgeGroup, Gender, condition, count(*) AS trials,
               round(avg(duration_s), 2) AS mean_duration_s
        FROM trials WHERE condition <> 'ss'
        GROUP BY ALL ORDER BY AgeGroup, Gender, condition
    """).df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Peak plantar pressure and body mass index** (the peak of every walking trial, averaged per
    subject and speed, using the frame-level peak pressures).
    """)
    return


@app.cell
def _(COLORS, SPEED, con, plt):
    peak_bmi = con.sql("""
        WITH peaks AS (SELECT subject, condition, trial, max(peak_pressure) AS peak FROM pressure_features
                       WHERE condition IN ('ws', 'wc', 'wf') GROUP BY ALL)
        SELECT p.subject, p.condition, avg(p.peak) AS peak_kpa, s.BMI
        FROM peaks p JOIN subjects s USING (subject) GROUP BY ALL
    """).df()
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _c in ("ws", "wc", "wf"):
        _d = peak_bmi[peak_bmi.condition == _c]
        _ax.scatter(_d.BMI, _d.peak_kpa, s=14, color=COLORS[_c], label=SPEED[_c])
    _ax.set_xlabel("Body mass index (kg/m²)")
    _ax.set_ylabel("Mean peak pressure (kPa)")
    _ax.legend(title="Speed", frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **4. Pressure-derived vertical force throughout the fast-walking trials of older women**, including unloaded frames.
    """)
    return


@app.cell
def _(COLORS, con, plt):
    force = con.sql("""
        SELECT f.trial_id, f.frame / t.frequency_hz AS time_s, f.grf_z AS force_n
        FROM pressure_features f JOIN trials t USING (trial_id)
        WHERE t.AgeGroup = 'O' AND t.Gender = 'F' AND t.condition = 'wf'
        ORDER BY f.trial_id, f.frame
    """).df()
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    for _tid, _g in force.groupby("trial_id"):
        _ax.plot(_g.time_s, _g.force_n, color=COLORS["wf"], lw=0.5, alpha=0.4)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Total vertical force (N)")
    _ax.set_title(f"{force.trial_id.nunique()} fast-walking trials of the older women", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **5. Peak vertical ground reaction force of each foot by task**

    For each foot, average the trial peaks across trials in which that foot loaded the
    platform. Express each participant's mean in body weights, then average across
    contributing participants. The table gives the participant count for each foot.
    """)
    return


@app.cell
def peak_force_by_task(con):
    con.sql("""
        WITH trial_peaks AS (SELECT subject, condition, trial,
                                    nullif(max(grf_z_left), 0) AS left_n, nullif(max(grf_z_right), 0) AS right_n
                             FROM pressure_features WHERE condition <> 'ss' GROUP BY ALL),
             subject_peaks AS (SELECT subject, condition, avg(left_n) AS left_n, avg(right_n) AS right_n FROM trial_peaks GROUP BY ALL)
        SELECT p.condition, count(p.left_n) AS left_subjects, count(p.right_n) AS right_subjects,
               round(avg(p.left_n / (s.Mass * 9.81)), 2) AS left_peak_bw,
               round(avg(p.right_n / (s.Mass * 9.81)), 2) AS right_peak_bw
        FROM subject_peaks p JOIN subjects s USING (subject)
        WHERE s.Mass > 0 GROUP BY p.condition ORDER BY p.condition
    """).df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **6. Area of the registered peak-pressure images** (area of positive interpolated pixels, grouped by walking speed and foot).
    """)
    return


@app.cell
def _(con):
    _area = con.sql("""
        WITH a AS (SELECT subject, condition, side, footprint, count(*) * 0.8469 * 0.8469 AS area_cm2 FROM footprints GROUP BY ALL)
        SELECT condition, side, round(avg(area_cm2), 1) AS mean_area_cm2, round(stddev(area_cm2), 1) AS sd_area_cm2, count(*) AS footprints
        FROM a GROUP BY ALL ORDER BY condition, side
    """).df()
    _area
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **7. How do contact time and vertical impulse vary by task and foot?**
    Each walking or running contact contributes one observation.
    """)
    return


@app.cell
def _(con):
    con.sql("""
        SELECT t.condition, c.side, count(*) AS contacts,
               round(avg(c.contact_time), 3) AS mean_duration_s,
               round(avg(c.vertical_impulse), 1) AS mean_impulse_n_s
        FROM contacts c JOIN trials t USING (trial_id)
        GROUP BY t.condition, c.side ORDER BY t.condition, c.side
    """).df()
    return


if __name__ == "__main__":
    app.run()
