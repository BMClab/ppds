# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "reader-data @ https://raw.githubusercontent.com/BMClab/ppds/796e7d1844209f46b787dc7a8b7442e3f8fe5961/reader_data-0.1.0-py3-none-any.whl",
#     "ppds @ https://raw.githubusercontent.com/BMClab/ppds/796e7d1844209f46b787dc7a8b7442e3f8fe5961/ppds-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPDS: example of 2D segmentation & registration

    > Laboratory of Biomechanics and Motor Control ([BMClab](https://github.com/BMClab)), Federal University of ABC, Brazil

    This notebook uses a running trial to demonstrate connected-component segmentation and
    registration of 2D plantar pressure images by centroid and principal axis. To explore the
    released walking images and their source contacts, see the registered footprints example.
    """)
    return


@app.cell
def _():
    from math import atan2, cos, pi, radians, sin

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage

    from ppds import dataset as ds
    from ppds.playback import pressure_player

    ds.use_notebook(mo.cli_args().get("zenodo"), pressure_sample=True)

    return atan2, cos, ds, mo, ndimage, np, pi, plt, pressure_player, radians, sin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: load data

    We will use trial 1 of subject s001 running at comfortable speed from the data set
    (`data/pressure.parquet`), loaded with `ds.pressure`, which returns the pressure of every frame
    as an array (frames, 353, 64) in kPa. Because the Zebris resolution is 0.1 N/cm² = 1 kPa, the
    values are integers and are stored as 16-bit unsigned integers, saving memory. Here the array is
    kept in the dataset convention (frame, y, x).
    """)
    return


@app.cell
def _(ds):
    I3D = ds.pressure("s001", "rc", 1)  # (frames, 353, 64), kPa
    I3D.shape
    return (I3D,)


@app.cell(hide_code=True)
def _(I3D, ds, pressure_player):
    pressure_player(I3D, ds.FREQ_HZ, title="s001 · Comfortable running · Trial 1")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create and visualize a 2D image (the maximum value of each sensor over time, the
    "peak pressure" image); 2D is easier to work with than 3D. Since we're going to be doing lots of
    image plotting with `imshow`, which accepts many keyword arguments that we'll use repeatedly, we
    create a simple `myimshow` function. In the transposed platform maps, the direction of
    travel (y) runs to the right and the transverse coordinate (x) increases downward.
    """)
    return


@app.cell
def _(I3D, np, plt):
    def myimshow(ax, I, *, transpose=False, colorbar=True, colorbar_orientation="horizontal"):
        """imshow with zeros masked (transparent) and a colorbar."""
        image = np.ma.masked_equal(I, 0)
        im = ax.imshow(image.T if transpose else image, interpolation="nearest",
                       origin="upper" if transpose else "lower")
        if colorbar:
            plt.colorbar(im, ax=ax, orientation=colorbar_orientation)
        return ax


    I = I3D.max(axis=0)  # maximum values by pixel, or "peak pressure" image
    _fig, _ax = plt.subplots(figsize=(10, 3))
    myimshow(_ax, I, transpose=True)
    _ax.set(xlabel="Along platform · y (sensor)", ylabel="Across · x (sensor)")
    _fig
    return I, myimshow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: segmentation

    In this case the feet are spaced far apart and there are zeros between the feet, so segmentation
    can be achieved easily using standard binary labeling:
    """)
    return


@app.cell
def _(I, myimshow, ndimage, plt):
    labels, n_labels = ndimage.label(I > 0)  # label image and number of labels
    _fig, _ax = plt.subplots(figsize=(10, 3))
    myimshow(_ax, labels, transpose=True)
    _ax.set_title(f"{n_labels} features")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the feet have different integer labels, it is easy to write a simple function that segments
    the feet into separate images:
    """)
    return


@app.cell
def _(I, myimshow, ndimage, np, plt):
    def segment(I):
        """One image per labeled feature."""
        L, n = ndimage.label(I > 0)
        II = []
        for i in range(n):
            a = I.copy()
            a[L != i + 1] = 0  # set all other values to zero
            II.append(a)
        return II


    II = segment(I)
    _fig, _ax = plt.subplots(figsize=(10, 4))
    myimshow(_ax, np.hstack(II), transpose=True)
    _fig
    return (II,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: registration

    For the rest of this notebook we will consider only two images, the first and the second right
    foot contacts. These two images will be our registration template and source, respectively,
    where "source" is the image that will be transformed to match the template.
    """)
    return


@app.cell
def _(II):
    I0 = II[0]  # first right foot contact (template)
    I1 = II[2]  # second right foot contact (source)
    return I0, I1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The registration procedure described below is a simple principal axis registration. It usually
    works decently on PP images from a single individual, but sometimes it does not. For general
    registration between individuals, more sophisticated methods are needed.

    To facilitate principal axis registration let's define a few functions:

    **Foot characteristic functions**:
    - `foot_coordinates` (coordinates of the nonzero pixels)
    - `centroid` (foot centroid)
    - `principal_axis` (orientation of the foot's long principal axis)

    **Image transformation functions**:
    - `transformXYR2D` (rigid 2D transformation with parameters $\delta x$, $\delta y$, $\delta \theta$)
    - `center` (center the foot in the current image grid)

    **Registration functions**:
    - `register_centroid` (register images so that their centroids align)
    - `register_pa` (register images so that their centroids and principal axes align)

    ### Foot characteristic functions
    """)
    return


@app.cell
def _(atan2, np):
    def foot_coordinates(I, th=0):
        """Coordinates of pixels with pressure above the threshold."""
        y, x = np.where(I > th)
        return x, y


    def centroid(I, th=0):
        """Pressure-weighted centroid of the selected sensors."""
        x, y = foot_coordinates(I, th)
        if not x.size:
            raise ValueError("Centroid requires at least one pixel above threshold")
        w = I[y, x]
        w = w / float(w.sum())
        return (w * x).sum(), (w * y).sum()


    def principal_axis(I, th=0):
        """Angle in radians of the undirected major axis from +x toward +y."""
        x, y = foot_coordinates(I, th)
        w = I[y, x]
        xc, yc = centroid(I, th)
        dx, dy = x - xc, y - yc
        xx = (w * dx**2).sum()
        yy = (w * dy**2).sum()
        xy = (w * dx * dy).sum()
        return 0.5 * atan2(2 * xy, xx - yy)

    return centroid, foot_coordinates, principal_axis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's see how these functions work:
    """)
    return


@app.cell
def _(I0, centroid, cos, foot_coordinates, plt, principal_axis, sin):
    _x, _y = foot_coordinates(I0)
    _xc, _yc = centroid(I0)
    _phi = principal_axis(I0)
    _fig, _ax = plt.subplots(figsize=(4, 6))
    _ax.plot(_x, _y, "bo", ms=3, label="Foot pixels")
    _ax.plot(_xc, _yc, "ro", ms=15, label="Centroid")
    _ax.plot([_xc, _xc + 10 * cos(_phi)], [_yc, _yc + 10 * sin(_phi)], "r-", lw=2, label="Principal axis")
    _ax.legend()
    _ax.axis("equal")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image transformation functions
    """)
    return


@app.cell
def _(I0, centroid, cos, myimshow, ndimage, np, plt, sin):
    def transformXYR2D(I, q, c=None):
        """Rigid 2D image transformation with parameters q = (dx, dy, dtheta).
        c is the rotation center (default: image center)."""

        def shift_fn(x, c0, q0):
            """Coordinate transformation function given parameters."""
            c = c0[1], c0[0]
            q = -q0[1], -q0[0], q0[2]
            cq3, sq3 = cos(q[2]), sin(q[2])
            xp0 = c[0] + q[0] * cq3 - q[1] * sq3 + x[0] * cq3 - x[1] * sq3 - c[0] * cq3 + c[1] * sq3
            xp1 = c[1] + q[0] * sq3 + q[1] * cq3 + x[0] * sq3 + x[1] * cq3 - c[0] * sq3 - c[1] * cq3
            return xp0, xp1

        if c is None:
            c = 0.5 * I.shape[1], 0.5 * I.shape[0]
        return ndimage.geometric_transform(I, shift_fn, order=1, extra_arguments=(c, q))


    def center(I):
        """Center the image on the current image grid."""
        x, y = centroid(I)
        m, n = I.shape
        return transformXYR2D(I, (0.5 * n - x, 0.5 * m - y, 0))


    _fig, _ax = plt.subplots(figsize=(8, 4))
    myimshow(_ax, np.hstack([I0, center(I0)]), transpose=True)  # before and after centering
    _fig
    return center, transformXYR2D


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `transformXYR2D` function can be used for general rigid-body transformations like this:
    """)
    return


@app.cell
def _(I0, centroid, myimshow, plt, radians, transformXYR2D):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    myimshow(_ax, transformXYR2D(I0, [-10, 50, radians(60)], c=centroid(I0)), transpose=True)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Registration functions
    """)
    return


@app.cell
def _(centroid, myimshow, np, pi, plt, principal_axis, transformXYR2D):
    def register_centroid(I0, I1, binary=False):
        """Register images so that their centroids align."""
        B0, B1 = (I0 > 0, I1 > 0) if binary else (I0, I1)
        xc0, yc0 = centroid(B0)
        xc1, yc1 = centroid(B1)
        return transformXYR2D(I1, (xc0 - xc1, yc0 - yc1, 0))


    def register_pa(I0, I1, binary=False):
        """Register images so that their centroids and principal axes align."""
        B0, B1 = (I0 > 0, I1 > 0) if binary else (I0, I1)
        xc0, yc0 = centroid(B0)
        xc1, yc1 = centroid(B1)
        dphi = principal_axis(B0) - principal_axis(B1)
        # A principal axis is unchanged by a half-turn. Use the smaller rotation
        # and pivot about the source centroid so the translation aligns centers.
        theta = (dphi + pi / 2) % pi - pi / 2
        return transformXYR2D(I1, (xc0 - xc1, yc0 - yc1, theta), c=(xc1, yc1))


    def plot_registration(template, source, registered, ylim=None, transpose=False):
        """Template, source and registered images side by side; gray contour = template outline."""
        II0 = np.hstack([template] * 3)
        II1 = np.hstack([template, source, registered])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contour((II0.T if transpose else II0) > 0, 1, colors="0.8", linewidths=3)
        myimshow(ax, II1, transpose=transpose)
        if ylim:
            ax.set_ylim(ylim)
        return fig

    return plot_registration, register_centroid, register_pa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first try `register_centroid`:
    """)
    return


@app.cell
def _(I0, I1, plot_registration, register_centroid):
    plot_registration(I0, I1, register_centroid(I0, I1), transpose=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The source image's centroid has been aligned with the template's centroid, but there isn't perfect
    overlap. The `register_pa` function can do slightly better (here both images are centered first):
    """)
    return


@app.cell
def _(I0, I1, center, plot_registration, register_pa):
    I0c, I1c = center(I0), center(I1)
    plot_registration(I0c, I1c, register_pa(I0c, I1c), ylim=(150, 200))
    return I0c, I1c


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `register_pa` can generally do even better if the pressure distribution is ignored and only the
    binary shapes are registered, using the `binary` keyword argument:
    """)
    return


@app.cell
def _(I0c, I1c, plot_registration, register_pa):
    I1r = register_pa(I0c, I1c, binary=True)
    plot_registration(I0c, I1c, I1r, ylim=(150, 200))
    return (I1r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Last, image transformations tend to leave artifacts at the foot periphery (due partly to the low
    image resolution), so it is generally best to apply a gentle threshold to transformed images:
    """)
    return


@app.cell
def _(I0c, I1r, myimshow, np, plt):
    _I1t = np.where(I1r < 5, 0, I1r)
    _fig, _ax = plt.subplots(figsize=(4, 5))
    _ax.contour(I0c > 0, 1, colors="0.8", linewidths=3)  # template outline
    myimshow(_ax, _I1t)
    _ax.set_ylim(150, 200)
    _fig
    return


if __name__ == "__main__":
    app.run()
