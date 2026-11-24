"""Interactive playback of one PPDS pressure recording in a marimo cell."""

import base64
import html
import json
from importlib.resources import files
from string import Template

import marimo as mo
import numpy as np
from matplotlib import colormaps

from .dataset import CELL_CM


def pressure_player(pressure, frequency_hz, *, title, stance=False, first_frame=0):
    """Play a uint16 kPa array indexed [frame, y, x] on its original trial clock.

    Pass first_frame when playing a slice from a trial. The view encloses all
    loaded sensors in the selection and keeps a common spatial and color scale.
    """
    pressure = np.asarray(pressure)
    if pressure.ndim != 3 or pressure.dtype != np.uint16 or 0 in pressure.shape:
        raise ValueError(
            "Expected a nonempty uint16 pressure array indexed [frame, y, x]"
        )
    rate = np.asanyarray(frequency_hz)
    if rate.ndim != 0 or rate.dtype.kind not in "iuf" or np.ma.is_masked(rate):
        raise ValueError("Expected a positive sampling frequency")
    hz = float(rate)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("Expected a positive sampling frequency")
    if isinstance(first_frame, np.ndarray) and first_frame.ndim == 0 and not np.ma.is_masked(first_frame):
        first_frame = first_frame[()]
    if (
        isinstance(first_frame, (bool, np.bool_))
        or not isinstance(first_frame, (int, np.integer))
        or first_frame < 0
    ):
        raise ValueError("Expected a nonnegative starting frame")

    frames, rows, cols = pressure.shape
    frame, y, x = np.nonzero(pressure)
    values = pressure[frame, y, x]
    # Screen coordinates increase rightward and downward. Gait advances to
    # the right. Stance retains the x-horizontal view of the stance plots.
    across, vertical = (x, y) if stance else (y, x)
    width, height = (cols, rows) if stance else (rows, cols)
    if values.size:
        left, right = max(0, across.min() - 2), min(width - 1, across.max() + 2)
        low, high = max(0, vertical.min() - 2), min(height - 1, vertical.max() + 2)
        width, height = int(right - left + 1), int(high - low + 1)
        screen_row = high - vertical if stance else vertical - low
        pixels = screen_row * width + across - left
    else:
        pixels = across
    if width * height > 65536:
        raise ValueError("Pressure display exceeds the 16-bit pixel index range")

    # One offset per frame keeps unloaded frames on the clock. Playback and
    # seeking then use only the selected trial's embedded sensor readings.
    offsets = np.r_[0, np.bincount(frame, minlength=frames).cumsum()].astype("<u4")
    readings = np.column_stack((pixels, values)).astype("<u2")
    packed = base64.b64encode(offsets.tobytes() + readings.tobytes()).decode("ascii")
    palette = colormaps["viridis"](np.linspace(0, 1, 256), bytes=True)
    config = {
        "packed": packed,
        "palette": base64.b64encode(palette.tobytes()).decode("ascii"),
        "frames": frames,
        "hz": hz,
        "first": int(first_frame),
        "width": width,
        "height": height,
        "peak": int(values.max(initial=0)),
        "newtonsPerKpa": 0.1 * CELL_CM**2,
    }
    template = Template(
        files(__package__).joinpath("pressure_player.html").read_text(encoding="utf-8")
    )
    document = template.substitute(
        title=html.escape(str(title)),
        axes="x → · y ↑" if stance else "y → · x ↓",
        config=json.dumps(config, separators=(",", ":")),
    )
    return mo.iframe(document, height="340px")
