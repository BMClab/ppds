"""Segment and register individual footprint images."""

from math import atan2, cos, pi, sin
from operator import index

import numpy as np
from scipy import ndimage


def segment(I, threshold=0, n_min=1, closing=0):
    """Label 8-connected pixels strictly above threshold, optionally closing the mask.

    Keep regions with at least n_min original above-threshold pixels.
    closing is the number of closing iterations (zero skips closing). For
    compatibility, boolean True uses two iterations; in 0.1.0 any truthy value,
    including 1, used two. Pixels added by closing connect regions but do not
    contribute to their size or pressure, and closing keeps every original
    above-threshold pixel, including those on the image border.
    """
    iterations = 2 * int(closing) if isinstance(closing, (bool, np.bool_)) else index(closing)
    if iterations < 0:
        raise ValueError("closing must be a nonnegative number of iterations")
    s = ndimage.generate_binary_structure(2, 2)  # features connected even if they touch diagonally
    active = I > threshold
    regions = active
    if iterations:
        # Pad so the erosion step does not treat the image border as background.
        padded = np.pad(active, iterations)
        closed = ndimage.binary_closing(padded, structure=s, iterations=iterations)
        regions = closed[iterations:-iterations, iterations:-iterations] | active
    labels, n = ndimage.label(regions, structure=s)
    counts = np.bincount(labels[active], minlength=n + 1)
    return [np.where((labels == i) & active, I, 0) for i in range(1, n + 1) if counts[i] >= n_min]


def foot_coordinates(I, threshold=0):
    """Coordinates (x, y) of the pixels above threshold."""
    y, x = np.where(I > threshold)
    return x, y


def centroid(I, threshold=0):
    """Pressure-weighted image centroid (xc, yc)."""
    x, y = foot_coordinates(I, threshold)
    if not x.size:
        raise ValueError("Centroid requires at least one pixel above threshold")
    w = I[y, x]
    w = w / float(w.sum())
    xc, yc = (w * x).sum(), (w * y).sum()
    return xc, yc


def principal_axis(I, threshold=0):
    """Undirected major-axis angle in radians from +x toward +y, in [-π/2, π/2].

    An isotropic image has no preferred direction and uses angle zero.
    """
    x, y = foot_coordinates(I, threshold)
    w = I[y, x]
    xc, yc = centroid(I, threshold)
    dx, dy = x - xc, y - yc
    xx = (w * dx**2).sum()
    yy = (w * dy**2).sum()
    xy = (w * dx * dy).sum()
    return 0.5 * atan2(2 * xy, xx - yy)


def _published_axis(mask):
    """Replay the original eigenvector calculation used for the released walking maps."""
    x, y = foot_coordinates(mask)
    w = mask[y, x]
    xc, yc = centroid(mask)
    # second moments
    J = np.eye(2)
    J[0, 0] = (w * (y - yc) ** 2).sum()
    J[1, 1] = (w * (x - xc) ** 2).sum()
    J[0, 1] = (w * (x - xc) * (y - yc)).sum()
    J[1, 0] = J[0, 1]
    # Preserve the release's row indexing and operation order for exact replay.
    d, v = np.linalg.eig(J)
    d, v = d.real, v.real
    vx, vy = v[d.argmin(), :]
    phi = pi / 2 - atan2(vx, vy)  # principal axis orientation
    return phi


def transformXYR2D(I, q, c=None):
    """Rigid 2D image transformation with parameters q = (dx, dy, dtheta).

    c is the rotation center (default: image center).
    """

    if c is None:
        c = 0.5 * I.shape[1], 0.5 * I.shape[0]
    cx, cy = c
    dx, dy, theta = q[:3]
    co, si = cos(theta), sin(theta)
    y, x = np.ogrid[:I.shape[0], :I.shape[1]]
    # Evaluate the original inverse coordinate map on the grid in one call.
    # Keep the arithmetic order to reproduce the scalar interpolation bit for bit.
    y0 = cy + (-dy) * co - (-dx) * si
    x0 = cx + (-dy) * si + (-dx) * co
    # A caller can supply e.g. float32 transform parameters. The scalar callback
    # combined them with Python floats using NumPy's scalar promotion rules.
    y_type, x_type = np.result_type(y0, 0.0), np.result_type(x0, 0.0)
    source_y = (
        y0 + (y * co).astype(y_type, copy=False) - (x * si).astype(y_type, copy=False)
        - cy * co + cx * si
    )
    source_x = (
        x0 + (y * si).astype(x_type, copy=False) + (x * co).astype(x_type, copy=False)
        - cy * si - cx * co
    )
    coordinates = np.array((source_y, source_x), dtype=np.float64)
    return ndimage.map_coordinates(I, coordinates, order=1)


def center(I, threshold=0):
    """Center image (its centroid) on the current image grid."""
    x, y = centroid(I, threshold)
    m, n = I.shape
    q = 0.5 * n - x, 0.5 * m - y, 0
    return transformXYR2D(I, q)


def register_centroid(I0, I1, binary=False, threshold=0):
    """Register I1 to I0 so that their centroids align."""
    if binary:
        B0, B1 = I0 > threshold, I1 > threshold
    else:
        B0, B1 = np.where(I0 > threshold, I0, 0), np.where(I1 > threshold, I1, 0)
    xc0, yc0 = centroid(B0, threshold=0)
    xc1, yc1 = centroid(B1, threshold=0)
    q = xc0 - xc1, yc0 - yc1, 0
    return transformXYR2D(I1, q)


def register_pa(I0, I1, binary=False, threshold=0):
    """Align centroids and major axes by rotating about the source centroid.

    Use the shortest rotation between the undirected axes, preserving the
    orientation of similarly directed feet. Interpolation and clipping to the
    source grid can change moments measured on the output raster.
    """
    if binary:
        B0, B1 = I0 > threshold, I1 > threshold
    else:
        B0, B1 = np.where(I0 > threshold, I0, 0), np.where(I1 > threshold, I1, 0)
    xc0, yc0 = centroid(B0, threshold=0)
    xc1, yc1 = centroid(B1, threshold=0)
    phi0 = principal_axis(B0)
    phi1 = principal_axis(B1)
    theta = (phi0 - phi1 + pi / 2) % pi - pi / 2
    return transformXYR2D(I1, (xc0 - xc1, yc0 - yc1, theta), c=(xc1, yc1))


def register_published_footprint(template, centered):
    """Reproduce PPDS walking registration on two pressure-centered kPa images.

    Footprint zero supplies the template. The release rotated subsequent
    binary silhouettes about the grid center using its original principal-axis
    calculation. Keep both conventions here to reproduce the stored pixels.
    """
    target_shape, source_shape = template > 0, centered > 0
    xc0, yc0 = centroid(target_shape)
    xc1, yc1 = centroid(source_shape)
    theta = _published_axis(target_shape) - _published_axis(source_shape)
    return transformXYR2D(centered, (xc0 - xc1, yc0 - yc1, theta))
