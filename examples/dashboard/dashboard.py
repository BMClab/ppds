# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/5188514abd44cdda18863454b96541f932c0ab13/notebook-assets/reader_data-0.2.1-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/5188514abd44cdda18863454b96541f932c0ab13/notebook-assets/ppds-0.2.1-py3-none-any.whl",
# ]
# ///

"""Run from the checkout with uv run --locked marimo run examples/dashboard/dashboard.py."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="PPDS · Pressure dashboard")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from dashboard_support import gait, plots
    from dashboard_support import measurements as data

    from ppds import dataset as ds

    # In the browser this reads Zenodo by byte range: each selection fetches
    # only the Parquet row groups of its trials.
    release = data.open_release()
    catalog = ds.trials()
    return catalog, data, ds, gait, mo, np, plots, release


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Plantar pressure dashboard
    """)
    return


@app.cell
def _(catalog, mo):
    _ids = sorted(catalog.subject.unique())
    subject = mo.ui.dropdown(_ids, value=_ids[0], searchable=True, label="Subject")
    scope = mo.ui.radio(
        ["One trial", "All trials"], value="One trial", inline=True, label="Show"
    )
    return scope, subject


@app.cell
def _(catalog, data, mo, subject):
    subject_trials = catalog.loc[catalog.subject.eq(subject.value)]
    _options = {
        label: code
        for label, code in data.CONDITIONS.items()
        if code in subject_trials.condition.values
    }
    condition = mo.ui.dropdown(_options, value=next(iter(_options)), label="Condition")
    return condition, subject_trials


@app.cell
def _(condition, mo, subject_trials):
    condition_trials = subject_trials.loc[
        subject_trials.condition.eq(condition.value)
    ].sort_values("trial")
    _options = {str(n): int(n) for n in condition_trials.trial}
    trial = mo.ui.dropdown(_options, value=next(iter(_options)), label="Trial")
    return condition_trials, trial


@app.cell(hide_code=True)
def _(condition, mo, scope, subject, trial):
    _controls = [subject, condition, scope]
    if scope.value == "One trial":
        _controls.append(trial)
    mo.hstack(_controls, justify="start", align="end", gap=2, wrap=True)
    return


@app.cell
def _(mo):
    def table(frame, *, select=False, first=False):
        if not select:
            return mo.Html(
                f'<div style="overflow-x:auto">{frame.to_html(index=False, border=0)}</div>'
            )
        return mo.ui.table(
            frame.reset_index(drop=True),
            selection="single",
            initial_selection=[0] if first and len(frame) else [],
            pagination=False,
            show_column_summaries=False,
            show_data_types=False,
            show_search=False,
            show_download=False,
        )

    return (table,)


@app.cell(hide_code=True)
def _(mo, subject_trials, table):
    _labels = {
        "subject": "Subject",
        "Age": "Age (yr)",
        "Gender": "Gender",
        "Height": "Height (cm)",
        "Mass": "Mass (kg)",
        "BMI": "BMI (kg/m²)",
        "DominantLeg": "Dominant leg",
    }
    _info = subject_trials.iloc[:1][list(_labels)].rename(columns=_labels)
    mo.vstack([mo.md("## Subject"), table(_info)])
    return


@app.cell
def _(condition, condition_trials, data, release, scope, trial):
    all_trials = scope.value == "All trials"
    number = None if all_trials else trial.value
    selected = (
        condition_trials
        if all_trials
        else condition_trials.loc[condition_trials.trial.eq(number)]
    )
    frames, contacts = data.trial_features(selected, release)
    standing = condition.value == "ss"
    walking = condition.value in ("ws", "wc", "wf")
    return all_trials, contacts, frames, selected, standing, walking


@app.cell
def _(
    all_trials,
    condition,
    condition_trials,
    contacts,
    data,
    frames,
    mo,
    selected,
    table,
):
    _summary = data.trial_summary(selected, frames, contacts)
    _can_highlight = all_trials and len(selected) > 1
    pass_table = table(_summary, select=_can_highlight, first=_can_highlight)
    _count = len(condition_trials)
    _total = f"{_count} recorded trial{'s' if _count != 1 else ''}"
    _scope = (
        f"All {_total}" if all_trials else f"Trial {selected.trial.iloc[0]} · {_total}"
    )
    _elements = [mo.md(f"## {condition.selected_key} · {_scope}")]
    if _can_highlight:
        _elements.append(
            mo.md(
                "Select a trial to highlight its walkway map, force interval, and centre-of-pressure paths."
            )
        )
    mo.vstack([*_elements, pass_table])
    return (pass_table,)


@app.cell
def _(all_trials, pass_table, selected):
    if all_trials and len(selected) > 1:
        _row = pass_table.value
        focus_trial = None if _row.empty else int(_row.Trial.iloc[0])
    else:
        focus_trial = int(selected.trial.iloc[0])
    return (focus_trial,)


@app.cell
def _(contacts, data, ds, np, release, selected, standing, subject):
    if standing:
        standing_mean = ds.standing_image(subject.value, kind="mean")
        peaks, cells, means = {}, None, {}
        _max = standing_mean.max()
    else:
        standing_mean = None
        peaks, cells, _registered = data.gait_maps(selected, release)
        means = data.mean_footprints(_registered, contacts)
        _max = max(image.max() for image in [*peaks.values(), *means.values()])
    vmax = max(50, 50 * np.ceil(_max / 50))
    return cells, means, peaks, standing_mean, vmax


@app.cell(hide_code=True)
def _(cells, focus_trial, mo, peaks, plots, standing, vmax):
    mo.stop(standing)
    mo.vstack(
        [
            mo.md("## Peak pressure along the walkway"),
            plots.walkway(peaks, cells, focus_trial, vmax),
        ]
    )
    return


@app.cell(hide_code=True)
def _(all_trials, contacts, data, frames, mo, selected, standing, table):
    if standing:
        _body = [mo.md("### Standing features"), table(data.standing_summary(frames))]
    elif all_trials:
        _body = [
            mo.md("### Contacts · all trials"),
            mo.md(
                "Mean ± sample SD across contacts for each foot. A single contact shows its value alone."
            ),
            table(data.contact_summary(contacts)),
        ]
    else:
        _body = [
            mo.md(f"### Contacts · trial {int(selected.trial.iloc[0])}"),
            table(data.contact_details(contacts))
            if len(contacts)
            else mo.md("No recorded contacts."),
        ]
    mo.vstack(_body)
    return


@app.cell
def _(data, ds, mo, plots, subject):
    scan = mo.ui.plotly(
        plots.plantar_scan({side: ds.scan(subject.value, side) for side in data.SIDES}),
        config={"displaylogo": False, "scrollZoom": False, "displayModeBar": False},
    )
    return (scan,)


@app.cell
def _(
    cells,
    contacts,
    means,
    mo,
    plots,
    standing,
    standing_mean,
    vmax,
    walking,
):
    if walking:
        detail = mo.vstack(
            [
                mo.md("### Mean registered peak pressure (kPa)"),
                plots.footprints(means, contacts.side.value_counts(), vmax),
            ]
        )
    elif standing:
        detail = mo.vstack(
            [
                mo.md("### Mean pressure during standing"),
                plots.standing_pressure(standing_mean, vmax),
            ]
        )
    else:
        detail = mo.vstack(
            [
                mo.md("### Contact peak pressure"),
                plots.contact_maps(cells, contacts, vmax),
                mo.md(
                    "Each foot shows its highest-peak contact in the selected trials."
                ),
            ]
        )
    return (detail,)


@app.cell(hide_code=True)
def _(detail, mo, scan):
    mo.hstack(
        [
            detail,
            mo.vstack(
                [
                    mo.md("### Foot scans"),
                    scan,
                    mo.md(
                        "The scans are displayed 5 cm apart. Drag to rotate both feet."
                    ),
                ]
            ),
        ],
        widths="equal",
        align="start",
        gap=2,
        wrap=True,
    )
    return


@app.cell
def _(cells, contacts, data, frames, gait, mo, standing, walking):
    mo.stop(standing)
    diagram = gait.cop_diagram(cells, contacts, frames, walking=walking)
    curves = data.stride_curves(frames, contacts)
    return curves, diagram


@app.cell(hide_code=True)
def _(curves, diagram, focus_trial, frames, mo, plots, selected):
    mo.vstack(
        [
            mo.md("## Gait forces and centre of pressure"),
            plots.gait_report(selected, frames, diagram, curves, focus_trial),
            mo.md(
                "Footprints show mean peak pressure after centring each contact. The blue butterfly path weights simultaneous foot COPs by their vertical forces. A walking pass is drawn from its first to its last double support on the walkway, because the first and last contacts share double support with a foot the walkway does not measure. The spacing between footprints is schematic. "
                "The bottom panels show mean stance force with ±1 sample SD across complete strides, ending at mean toe-off on the gait-cycle axis."
            ),
        ]
    )
    return


@app.cell
def _(frames, mo, plots, selected, standing):
    mo.stop(not standing)
    standing_grf, standing_cop = plots.standing_time_series(selected.iloc[0], frames)
    return standing_cop, standing_grf


@app.cell(hide_code=True)
def _(mo, standing_grf):
    mo.vstack([mo.md("## Vertical ground reaction force over time"), standing_grf])
    return


@app.cell(hide_code=True)
def _(mo, standing_cop):
    mo.vstack([mo.md("## Centre of pressure over time"), standing_cop])
    return


if __name__ == "__main__":
    app.run()
