"""Regression checks for notebook selections on reduced PPDS releases."""

from types import SimpleNamespace

import duckdb
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from reader_data import Release

from ppds.paths import ROOT


@pytest.mark.parametrize("cell", ["export", "round_trip"])
def test_conversion_cells_can_rerun_independently(monkeypatch, cell):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples import pressure_conversion as notebook

    _, source = notebook.import_ascii.run()
    inputs = {name: source[name] for name in ("ascii_tables", "work")}
    _, exported = notebook.export_npz.run(**inputs)
    if cell == "export":
        _, repeated = notebook.export_npz.run(**inputs)
        assert [p.name for p in repeated["archives"]] == [p.name for p in exported["archives"]]
        assert all(p.is_file() for p in repeated["archives"])
    else:
        for _ in range(2):
            notebook.check_round_trip.run(**inputs, archives=exported["archives"], trials=source["trials"])


def test_peak_force_averages_only_observed_foot_peaks(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.queries import peak_force_by_task

    with duckdb.connect() as con:
        con.execute("""
            CREATE TABLE subjects AS SELECT * FROM (VALUES ('a', 100), ('b', 100), ('c', NULL)) AS t(subject, Mass)
        """)
        con.execute("""
            CREATE TABLE pressure_features AS SELECT * FROM (VALUES
                ('a', 'wc', 1, 981., 981.), ('a', 'wc', 2, 0., 981.),
                ('b', 'wc', 1, 0., 1962.), ('c', 'wc', 1, 981., 981.)
            ) AS t(subject, condition, trial, grf_z_left, grf_z_right)
        """)
        result, _ = peak_force_by_task.run(con=con)
    row = result.iloc[0]
    assert (row.left_peak_bw, row.right_peak_bw) == (1., 1.5)
    assert (row.left_subjects, row.right_subjects) == (1, 2)


@pytest.fixture
def explorer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples import exploration

    return exploration


def test_characteristic_filter_groups_equivalent_responses(explorer):
    people = pd.DataFrame({"Footwear": ["Sneakers", " Sneakers ", "", "  ", None]})
    choice = mo.ui.dropdown({"Usual footwear": "Footwear"}, value="Usual footwear")
    _, selection = explorer.response_filter.run(people=people, characteristic=choice, mo=mo)
    assert list(selection["category"].options) == ["All responses", "Missing response", "Sneakers"]
    assert selection["responses"].value_counts().to_dict() == {"Missing response": 3, "Sneakers": 2}


@pytest.mark.parametrize("sides", ["L", "R", "LR", ""])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_median_footprints_use_available_maps(explorer, sides):
    def footprints(subject, condition, side):
        maps = np.zeros((int(side in sides), 353, 64))
        if len(maps):
            maps[:, 201:205, 7:9] = 300
        return maps

    _, plotting = explorer.footprint_plot.run(np=np, plt=plt)
    output, _ = explorer.median_footprints.run(
        condition=SimpleNamespace(value="wf"), participant=SimpleNamespace(value="s002"),
        show_medians=SimpleNamespace(value=True), ds=SimpleNamespace(footprints=footprints),
        mo=mo, np=np, peak_panels=plotting["peak_panels"],
    )
    if not sides:
        assert "no registered footprints" in output.text.lower()
        return
    try:
        panels = [axis for axis in output.axes if axis.images]
        assert [axis.get_title().split(" · ")[0] for axis in panels] == [
            name for side, name in (("L", "Left"), ("R", "Right")) if side in sides
        ]
        for axis in panels:
            assert axis.images[0].get_clim() == (0, 300)
            low, high = axis.get_ylim()
            assert low <= 201 and high >= 204
            assert high - low < 20
    finally:
        plt.close(output)


@pytest.mark.parametrize("only_stance", [False, True])
def test_pressure_notebook_selects_available_gait(data_root, monkeypatch, only_stance):
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.pressure import app

    # Model a smaller release by limiting the trial inventory used by every reader and SQL join.
    table = Release.table
    selected = "subject = 's001' AND condition = 'ss'"
    if not only_stance:
        selected += " OR (subject = 's002' AND condition = 'wf' AND trial = 2)"

    def subset(release, name, *, connection=None):
        rows = table(release, name, connection=connection)
        return rows.filter(selected) if name == "trials" else rows

    monkeypatch.setattr(Release, "table", subset)
    try:
        outputs, defs = app.run()
        if only_stance:
            assert "p" not in defs
            assert any("no walking or running" in getattr(o, "text", "").lower() for o in outputs)
        else:
            assert list(defs["subject"].options) == ["s002"]
            assert defs["condition"].value == "wf"
            assert defs["trial"].value == "2"
            assert defs["p"].size > 0
            frame_plot = next(
                o for o in outputs if hasattr(o, "axes") and o.axes[0].get_xlabel() == "y (sensor)"
            )
            assert frame_plot.axes[0].yaxis_inverted()
    finally:
        plt.close("all")


def test_subjects_and_trials_query_handles_bad_queries_without_hiding_io_errors(monkeypatch):
    import tokenize

    monkeypatch.syspath_prepend(str(ROOT))
    from examples.subjects_and_trials import query_trials

    people = pd.DataFrame({"subject": ["s001"], "Age": [30.0]})
    stub = SimpleNamespace(
        subjects=lambda q=None: people.query(q) if q else people,
        trials=lambda **kwargs: pd.DataFrame({"subject": ["s001"]}),
    )
    for bad in ("NoSuchCol > 0", "Age ==", "Age == 'unterminated"):
        output, _ = query_trials.run(
            ds=stub, mo=mo, pd=pd, query=SimpleNamespace(value=bad),
            task=SimpleNamespace(value=None), tokenize=tokenize,
        )
        assert "Invalid query" in output.text

    broken = SimpleNamespace(
        subjects=lambda q=None: (_ for _ in ()).throw(OSError("disk unavailable")),
        trials=lambda **kwargs: pd.DataFrame(),
    )
    with pytest.raises(OSError, match="disk unavailable"):
        query_trials.run(
            ds=broken, mo=mo, pd=pd, query=SimpleNamespace(value="Age > 0"),
            task=SimpleNamespace(value=None), tokenize=tokenize,
        )
