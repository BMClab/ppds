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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3D scans of the feet and lower legs

    The [Laboratory of Biomechanics and Motor Control](https://github.com/BMClab)
    at the Federal University of ABC, Brazil, scanned each foot and lower leg separately with
    a Microsoft Kinect sensor and Skanect 1.8.3. The release contains binary STL meshes in
    metres for 95 participants. Choose a participant to explore both anatomical sides. Drag either mesh to rotate it
    and scroll to zoom.

    Each panel shows the original scan axes in centimetres. The starting views use the
    same scale. Position and orientation can differ because each scan has its own
    coordinate system.
    """)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from ppds import dataset as ds

    ds.use_notebook(mo.cli_args().get("zenodo"))

    return ds, go, make_subplots, mo, np, pd


@app.cell
def _(ds, mo):
    _participants = ds.subjects().subject.tolist()
    participant = mo.ui.dropdown(
        options=_participants, value=_participants[0], label="Participant"
    )
    mo.output.replace(participant)
    return (participant,)


@app.cell
def _(ds, participant):
    # ds.scan returns vertex positions in metres and indices for every triangle.
    meshes = {
        name: ds.scan(participant.value, side)
        for name, side in [("Left", "l"), ("Right", "r")]
    }
    return (meshes,)


@app.cell
def _(go, make_subplots, meshes, mo, np):
    _fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"{name} foot and lower leg" for name in meshes],
        horizontal_spacing=0.02,
    )
    # Centre each panel on its mesh while preserving the original coordinates.
    # Identical axis spans and orthographic cameras give both views the same scale.
    _span_cm = 1.1 * np.max([np.ptp(v * 100, axis=0) for v, _ in meshes.values()], axis=0)
    _aspect = dict(zip("xyz", _span_cm / np.cbrt(_span_cm.prod())))
    for _column, (_side, (_vertices, _faces)) in enumerate(meshes.items(), start=1):
        _cm = _vertices * 100
        _x, _y, _z = _cm.T
        _i, _j, _k = _faces.T
        _fig.add_trace(
            go.Mesh3d(
                x=_x, y=_y, z=_z, i=_i, j=_j, k=_k, name=_side,
                color="#b0d0df", showscale=False,
                lighting={"ambient": 0.30, "diffuse": 0.68, "specular": 0.12, "roughness": 0.75},
                lightposition={"x": -30000, "y": 80000, "z": -50000},
                hovertemplate="x %{x:.1f} cm<br>y %{y:.1f} cm<br>z %{z:.1f} cm<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=_column,
        )
        _mid_cm = (_cm.min(axis=0) + _cm.max(axis=0)) / 2
        _axes = {
            f"{axis}axis": {
                "title": f"{axis} (cm)",
                "range": [mid - span / 2, mid + span / 2],
            }
            for axis, mid, span in zip("xyz", _mid_cm, _span_cm)
        }
        _fig.update_scenes(
            **_axes,
            aspectmode="manual",
            aspectratio=_aspect,
            camera={
                "eye": {"x": 1.2, "y": -1.6, "z": 0.9},
                "projection": {"type": "orthographic"},
            },
            dragmode="orbit",
            row=1,
            col=_column,
        )
    _fig.update_layout(height=405, margin={"l": 0, "r": 0, "b": 0, "t": 40})
    mo.ui.plotly(_fig, config={"scrollZoom": True, "displaylogo": False})


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mesh geometry

    Surface area is the sum of the triangle areas over the entire scanned surface.
    Native extents measure the span along each original scan axis. Principal extents
    measure the spans along the three principal axes of the vertices, ordered by
    decreasing variance. These dimensions describe the foot and the scanned portion
    of the lower leg together.
    """)


@app.cell
def _(meshes, np, pd):
    _rows = []
    for _side, (_vertices, _faces) in meshes.items():
        _cm = _vertices.astype(float) * 100
        _a, _b, _c = (_cm[_faces[:, k]] for k in range(3))
        _area = 0.5 * np.linalg.norm(np.cross(_b - _a, _c - _a), axis=1).sum()
        _centered = _cm - _cm.mean(axis=0)
        _, _vectors = np.linalg.eigh(np.cov(_centered.T))
        _principal = np.ptp(_centered @ _vectors[:, ::-1], axis=0)
        _rows.append({
            "Side": _side,
            "Vertices": len(_cm),
            "Triangles": len(_faces),
            "Surface area (cm²)": round(_area, 1),
            "x, y, z extents (cm)": ", ".join(f"{v:.1f}" for v in np.ptp(_cm, axis=0)),
            "Principal extents 1, 2, 3 (cm)": ", ".join(f"{v:.1f}" for v in _principal),
        })
    pd.DataFrame(_rows)


if __name__ == "__main__":
    app.run()
