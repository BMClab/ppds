# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/2b7fe938caa4b18b7082d20d50880858a93479bb/notebook-assets/reader_data-0.1.0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/2b7fe938caa4b18b7082d20d50880858a93479bb/notebook-assets/ppds-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from ppds import dataset as ds

    ds.use_notebook(mo.cli_args().get("zenodo"))

    return Line2D, MaxNLocator, ds, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Participant summary

    Each row from `ds.subjects()` represents one participant. Select a cohort to
    compare demographics, body and foot measurements, physical activity and
    self-reported health. Age groups follow the study's younger (Y) and older (O)
    adult classifications.
    """)


@app.cell
def _(ds):
    participants = ds.subjects()
    return (participants,)


@app.cell
def _(pd):
    # Labels and units follow the trials resource in the PPDS data dictionary.
    measures = {
        "Age": "Age (years)",
        "Height": "Height (cm)",
        "Mass": "Body mass (kg)",
        "BMI": "BMI (kg/m²)",
        "LegLength": "Average leg length (cm)",
        "FootLenR": "Right foot length (cm)",
        "FootLenL": "Left foot length (cm)",
        "FootWidR": "Right foot width (cm)",
        "FootWidL": "Left foot width (cm)",
        "FootwearSize": "Shoe size (Brazilian)",
        "PAfreq": "Physical activity (days/week)",
    }
    categories = {
        "AgeGroup": "Age group",
        "Gender": "Gender",
        "DominantLeg": "Dominant leg",
        "Footwear": "Usual footwear",
        "FootOrthosis": "Foot orthosis in footwear",
        "Ortho-Prosthesis": "Wears an orthosis or prosthesis",
        "PAtype": "Physical activity type",
        "Illness": "Current illness",
        "Medication": "Regular medication",
        "MSKinjury": "Musculoskeletal injury in past year",
        "Surgery": "Surgery in past year",
    }

    def responses(frame, column):
        values = frame[column].str.strip().replace("", pd.NA).fillna("Missing")
        if column == "AgeGroup":
            return values.replace({"Y": "Younger (Y)", "O": "Older (O)"})
        return values

    return categories, measures, responses


@app.cell
def _(mo, participants, responses):
    age_filter = mo.ui.dropdown(
        ["All", *sorted(responses(participants, "AgeGroup").unique())],
        value="All", label="Age group",
    )
    gender_filter = mo.ui.dropdown(
        ["All", *sorted(responses(participants, "Gender").unique())],
        value="All", label="Gender",
    )
    group_by = mo.ui.dropdown(
        {"Age group": "AgeGroup", "Gender": "Gender", "Whole cohort": None},
        value="Age group", label="Compare by",
    )
    mo.hstack([age_filter, gender_filter, group_by], justify="start", wrap=True)
    return age_filter, gender_filter, group_by


@app.cell
def _(age_filter, gender_filter, mo, participants, responses):
    cohort = participants
    for _column, _choice in [("AgeGroup", age_filter.value), ("Gender", gender_filter.value)]:
        if _choice != "All":
            cohort = cohort.loc[responses(cohort, _column).eq(_choice)]
    mo.md(f"**{len(cohort)} of {len(participants)} participants** match the filters.")
    return (cohort,)


@app.cell
def _(cohort, group_by, mo, pd, responses):
    mo.stop(cohort.empty, mo.md("Change a cohort filter to see the participant summaries."))
    _labels = (
        responses(cohort, group_by.value) if group_by.value
        else pd.Series("Whole cohort", index=cohort.index)
    )
    groups = cohort.groupby(_labels, sort=True)
    return (groups,)


@app.cell
def _(categories, mo):
    category = mo.ui.dropdown(
        {label: column for column, label in categories.items()},
        value="Gender", label="Characteristic",
    )
    mo.vstack([mo.md("## Participant counts"), category])
    return (category,)


@app.cell
def _(categories, category, groups, mo, pd, responses):
    _rows = []
    for _label, _group in groups:
        _counts = responses(_group, category.value).value_counts()
        if "Missing" not in _counts:
            _counts.loc["Missing"] = 0
        for _response, _n in _counts.items():
            _rows.append((_label, _response, _n, len(_group), 100 * _n / len(_group)))
    counts = pd.DataFrame(
        _rows, columns=["Group", categories[category.value], "Participants", "Group size", "% of group"],
    ).round({"% of group": 1})
    mo.vstack([
        mo.ui.table(counts, selection=None, page_size=10),
        mo.md("Percentages use everyone in each group, including missing responses. "
              "A reported activity combination counts as one response."),
    ])
    return (counts,)


@app.cell
def _(groups, measures, mo, pd):
    _tables = {}
    for _label, _group in groups:
        _stats = _group[list(measures)].describe().T.rename(columns={
            "count": "Observed", "mean": "Mean", "std": "SD", "min": "Min",
            "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max",
        })
        _stats["Observed"] = _stats["Observed"].astype(int)
        _stats.insert(1, "Missing", len(_group) - _stats["Observed"])
        _tables[_label] = _stats.rename(index=measures)
    summary = pd.concat(_tables, names=["Group", "Measure"]).reset_index().round(2)
    mo.vstack([
        mo.md("## Measurements"),
        mo.md("Observed and missing counts refer to participants. Q1 and Q3 are the 25th and 75th "
              "percentiles. The sample standard deviation (SD) requires at least two observed values."),
        mo.ui.table(summary, selection=None, page_size=len(measures)),
    ])
    return (summary,)


@app.cell
def _(measures, mo):
    _options = {label: column for column, label in measures.items()}
    x_measure = mo.ui.dropdown(_options, value=measures["Height"], label="Horizontal axis")
    y_measure = mo.ui.dropdown(_options, value=measures["FootLenR"], label="Vertical axis")
    mo.vstack([
        mo.md("## Distributions and relationships"),
        mo.hstack([x_measure, y_measure], justify="start", wrap=True),
    ])
    return x_measure, y_measure


@app.cell
def _(Line2D, MaxNLocator, cohort, groups, measures, np, plt, x_measure, y_measure):
    _x, _y = x_measure.value, y_measure.value
    _observed = cohort[_x].dropna()
    _fig, (_hist, _scatter) = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    if not _observed.empty:
        if _x in ("PAfreq", "FootwearSize"):
            _bins = np.arange(_observed.min() - 0.5, _observed.max() + 1.5)
            _hist.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            _bins = np.histogram_bin_edges(_observed, bins="auto")

    _legend = []
    for _i, (_label, _group) in enumerate(groups):
        _color = f"C{_i}"
        _style = ["-", "--", ":"][_i % 3]
        _marker = ["o", "^", "s"][_i % 3]
        if not _observed.empty:
            _hist.hist(_group[_x].dropna(), bins=_bins, histtype="step",
                       color=_color, linewidth=2, linestyle=_style)
        _pairs = _group.dropna(subset=[_x, _y])
        _scatter.scatter(_pairs[_x], _pairs[_y], color=_color, marker=_marker, alpha=0.75, s=26)
        _legend.append(Line2D([], [], label=_label, color=_color, linestyle=_style, marker=_marker))

    _n_pairs = len(cohort.dropna(subset=[_x, _y]))
    _hist.set(title=f"Distribution · {len(_observed)} observed", xlabel=measures[_x],
              ylabel="Participants")
    _hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    _scatter.set(title=f"Paired measurements · {_n_pairs} participants",
                 xlabel=measures[_x], ylabel=measures[_y])
    for _ax, _n in [(_hist, len(_observed)), (_scatter, _n_pairs)]:
        _ax.spines[["top", "right"]].set_visible(False)
        if _n == 0:
            _ax.text(0.5, 0.5, "No observations", ha="center", transform=_ax.transAxes)
    _fig.legend(handles=_legend, loc="outside lower center", ncols=3, frameon=False)
    _fig


@app.cell
def _(categories, category, cohort, measures, mo, x_measure, y_measure):
    _fields = {"subject": "Participant", **categories, **measures}
    _order = list(dict.fromkeys([
        "subject", "AgeGroup", "Gender", x_measure.value, y_measure.value, category.value, *_fields,
    ]))
    mo.vstack([
        mo.md("## Participant records"),
        mo.md("The selected measurements appear beside the participant ID and demographics. "
              "Sort a column to inspect individual values."),
        mo.ui.table(cohort[_order].rename(columns=_fields), selection=None, page_size=10),
    ])


if __name__ == "__main__":
    app.run()
