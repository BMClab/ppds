"""Per-sensor pressure maps and the strike index for one foot's frames.

Each function takes a [frame, y, x] pressure series in kPa, such as one gait
contact's frames from ``dataset.pressure(..., side=...)[start_frame:stop_frame]``
or a quiet-standing trial. Unloaded frames and cells count as zero pressure.
The PPDS 1.1.0 ``contact_maps`` and ``standing_maps`` tables store the peak and
pressure-time integral computed this way; the mean pressure follows from them.
"""

import numpy as np


def _frames(frames):
    frames = np.asarray(frames)
    if frames.ndim != 3 or not len(frames):
        raise ValueError("Expected a nonempty [frame, y, x] pressure series")
    if np.issubdtype(frames.dtype, np.bool_) or not np.issubdtype(frames.dtype, np.number):
        raise TypeError("Expected numeric pressures in kPa")
    return frames


def peak_pressure_map(frames):
    """Return each sensor's maximum pressure over the frames, in kPa."""
    return _frames(frames).max(axis=0)


def pressure_time_integral_map(frames, frequency_hz):
    """Return each sensor's pressure-time integral in kPa s: the frame sum divided by the sampling frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Expected a positive sampling frequency in Hz")
    return _frames(frames).sum(axis=0, dtype=np.float64) / frequency_hz


def mean_pressure_map(frames):
    """Return each sensor's mean pressure in kPa over all frames, unloaded frames included.

    This equals the pressure-time integral divided by the duration of the frames.
    """
    return _frames(frames).mean(axis=0, dtype=np.float64)


def strike_index(frames):
    """Return the strike index of one foot contact (Cavanagh and Lafortune, 1980).

    The center of pressure on the first frame, measured along y from the
    posterior edge of the contact footprint, as a fraction of the footprint
    length. The footprint spans the outer sensor edges of every cell loaded
    during the frames. Gait y increases in the walking direction, so 0 is the
    heel end and 1 the toe end. The first frame must carry load.
    """
    frames = _frames(frames)
    first = frames[0].astype(np.float64)
    load = first.sum()
    if load <= 0:
        raise ValueError("The first frame of a contact must carry load")
    if (frames < 0).any():
        raise ValueError("Pressures must be nonnegative")
    rows = np.nonzero(frames.max(axis=(0, 2)))[0]
    cop_y = (first.sum(axis=1) * np.arange(first.shape[0])).sum() / load
    return float((cop_y - (rows[0] - 0.5)) / (rows[-1] - rows[0] + 1))
