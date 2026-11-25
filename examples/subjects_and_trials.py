# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/00dea644fdc8dd42d7bc7e9a76925bed249453bc/notebook-assets/reader_data-0.1.1.dev0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/00dea644fdc8dd42d7bc7e9a76925bed249453bc/notebook-assets/ppds-0.1.1.dev0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: participants and trials

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    Participant characteristics appear on each trial row in `data/trials.parquet`.
    Use `ds.subjects()` for one row per participant, or filter trials by participant characteristics. Only the dataset folder `data/` is needed.

    ```python
    from ppds import dataset as ds
    ds.subjects()                                   # all participants
    ds.subjects("AgeGroup == 'O' and Gender == 'F'")  # selection (pandas query)
    ds.trials(condition="wf", query="Age > 60")     # fast-walking trials of the participants older than 60

    ds.trials(condition="ss")                       # one stance trial per participant
    ```
    """)
    return


@app.cell
def _():
    import tokenize

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from ppds import dataset as ds

    ds.use_notebook(mo.cli_args().get("zenodo"))

    CONDITIONS = {"ss": "quiet standing", "ws": "walking, slow", "wc": "walking, comfortable", "wf": "walking, fast", "rc": "running, comfortable"}
    plt.rcParams.update({"axes.grid": True, "grid.color": "0.9", "axes.spines.top": False, "axes.spines.right": False})
    return CONDITIONS, ds, mo, pd, plt, tokenize


@app.cell
def _(ds, mo):
    subjects = ds.subjects()
    trials = ds.trials()
    mo.md(f"**{len(subjects)}** participants and **{len(trials):,}** pressure trials.")
    return subjects, trials


@app.cell
def _(mo, subjects):
    mo.ui.table(subjects, selection=None, page_size=8, label="Participant characteristics")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Number of participants by age group (Y: 18–50 completed years, decimal `Age` 18.4–50.7; O: ≥ 59 years, decimal `Age` 59.2–87.1) and gender, and their main characteristics:
    """)
    return


@app.cell
def _(mo, pd, subjects):
    mo.hstack([
        pd.crosstab(subjects.AgeGroup, subjects.Gender, margins=True),
        subjects.groupby("AgeGroup")[["Age", "Height", "Mass", "BMI"]].agg(["mean", "std"]).round(1),
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select participants and trials

    Write a [pandas query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
    on the columns of the participant table (e.g. `Age > 60`, `Gender == 'F' and BMI < 25`,
    `FootLenR > 25`) and choose a task:
    """)
    return


@app.cell
def _(CONDITIONS, mo):
    query = mo.ui.text(value="AgeGroup == 'O' and Gender == 'F'", label="Subjects", full_width=True)
    task = mo.ui.dropdown(options={"all tasks": None, **{v: k for k, v in CONDITIONS.items()}}, value="walking, fast", label="Task")
    mo.vstack([query, task])
    return query, task


@app.cell
def query_trials(ds, mo, pd, query, task, tokenize):
    try:
        selected = ds.subjects(query.value or None)
        selected_trials = ds.trials(subject=selected.subject.tolist(), condition=task.value)
        _out = mo.vstack([
            mo.md(f"**Participants contributing trials** {selected_trials.subject.nunique()} of "
                  f"{len(selected)} matching the query · **Selected trials** {len(selected_trials)}"),
            mo.ui.table(selected_trials, selection=None, page_size=8),
        ])
    except (SyntaxError, KeyError, ValueError, TypeError, tokenize.TokenError, pd.errors.UndefinedVariableError) as _e:
        _out = mo.callout(mo.md(f"Invalid query: `{_e}`"), kind="danger")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data availability

    The last columns of the participant table give the number of pressure trials per task.
    `TrialsStanding` is 1 for every retained participant: one standing trial includes both feet.
    All participants have scans of both feet.
    Participants with fewer pressure trials than the protocol:
    """)
    return


@app.cell
def _(mo, subjects):
    _avail = ["TrialsWS", "TrialsWC", "TrialsWF", "TrialsRC", "TrialsStanding"]
    _ran = subjects.TrialsRC > 0
    _missing = ((subjects[["TrialsWS", "TrialsWC", "TrialsWF"]] < 10).any(axis=1) | (_ran & (subjects.TrialsRC < 10))
                | (subjects.TrialsStanding < 1))
    mo.ui.table(subjects.loc[_missing, ["subject", *_avail]], selection=None, page_size=20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trial duration by task

    Walking and running trials have variable duration, calculated from the recorded
    frame count at 100 Hz. Each quiet-standing trial uses the final 10 seconds of the recording.
    """)
    return


@app.cell
def _(CONDITIONS, plt, trials):
    _order = [c for c in CONDITIONS if c in set(trials.condition)]
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.boxplot([trials.loc[trials.condition == c, "duration_s"] for c in _order], tick_labels=[CONDITIONS[c] for c in _order],
                medianprops={"color": "0.2"}, patch_artist=True, boxprops={"facecolor": "0.85", "color": "0.4"})
    _ax.set_ylabel("Trial duration (s)")
    _ax.set_title(", ".join(f"{CONDITIONS[c]}: {(trials.condition == c).sum()} trials" for c in _order), fontsize=8)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
