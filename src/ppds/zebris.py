"""Read an exported Zebris gait pressure recording one exercise at a time."""

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np

MARGIN = 20  # sensor rows (20 * 0.85 cm) kept as margin when deleting a partial step (high arches)


def clean_zebris_trial(data, trial=1):
    """Apply the ASCII converter's boundary cleanup to one trial, then rotate its axes.

    Input axes are (ASCII row, ASCII column, frame); output is (frame, y, x).
    The caller's array is preserved. Returns (cleaned pressure, boundary warnings).
    This does not remove lateral-edge contacts or infer vendor contact boundaries.
    """
    cleaned, warnings, _ = _clean_ascii_trial(data, trial)
    return cleaned, warnings


def _clean_ascii_trial(data, trial):
    """Internal cleanup retaining the acquisition-frame offset through every trim."""
    data = np.array(data, dtype=float, copy=True)
    log = []
    frame_start = 0

    def del_zeros(data):
        nonlocal frame_start
        inot0 = np.flatnonzero(np.any(data, axis=(0, 1)))
        if not len(inot0):
            return data[:, :, :0]
        if inot0[0] == 0:  # trial started with subject already on the platform: delete first step
            log.append((trial, "Trial started with subject already on the platform."))
            loaded = data.any(axis=(0, 2))
            first = np.flatnonzero(loaded)[0]
            gaps = np.flatnonzero(~loaded[first:])
            end = first + gaps[0] if gaps.size else len(loaded)
            data[:, : end + MARGIN, :] = 0
        if inot0[-1] == data.shape[2] - 1:  # trial ended with subject still on the platform: delete last step
            log.append((trial, "Trial ended with subject still on the platform."))
            if data[:, :, -1].any():  # the entry trim may already have removed this contact
                loaded = data.any(axis=(0, 2))
                last = np.flatnonzero(loaded)[-1]
                gaps = np.flatnonzero(~loaded[:last])
                start = max(0, gaps[-1] - MARGIN) if gaps.size else 0
                data[:, start:, :] = 0
        if inot0[0] == 0 or inot0[-1] == data.shape[2] - 1:
            inot0 = np.where(np.sum(data, (0, 1)))[0]
        if not len(inot0):
            return data[:, :, :0]
        start = max(0, inot0[0] - 1)
        frame_start += start
        return data[:, :, start : inot0[-1] + 2]  # leave one zero frame

    data = del_zeros(data)
    if not data.shape[2]:
        return np.moveaxis(np.rot90(data, k=3), -1, 0), log, frame_start
    # delete step if subject stepped on the border
    if np.sum(data[:, 0, :]) > 0:
        log.append((trial, "Subject stepped on the inferior border."))
        gaps = np.flatnonzero(~data.any(axis=(0, 2)))
        end = gaps[0] if gaps.size else data.shape[1]
        data[:, :end, :] = 0
        data = del_zeros(data)
    if np.sum(data[:, -1, :]) > 0:
        log.append((trial, "Subject stepped on the superior border."))
        gaps = np.flatnonzero(~data.any(axis=(0, 2)))
        start = gaps[-1] if gaps.size else 0
        data[:, start:, :] = 0
        data = del_zeros(data)
    return np.moveaxis(np.rot90(data, k=3), -1, 0), log, frame_start


@contextmanager
def open_zebris_trials(fname, subject=None):
    """Open a gait pressure ASCII export and stream (grid, warnings, exercise timing).

    The iterator reads one exercise at a time, applies the notebook's boundary
    cleanup and returns (frame, y, x) grids in the units declared by the export.
    """
    fname = Path(fname)
    with fname.open("rb") as stream:
        def line():
            text = stream.readline()
            if not text:
                raise ValueError(f"{fname.name}: unexpected end of pressure export")
            text = text.rstrip(b"\r\n")
            try:
                return text.decode("utf-8")
            except UnicodeDecodeError:
                return text.decode("latin-1")

        def field():
            fields = line().split("\t")
            if len(fields) < 2:
                raise ValueError(f"{fname.name}: expected a tab-separated header field")
            return fields[1]

        def seek(marker):
            while line().strip() != marker:
                pass

        for _ in range(3):  # Do not retain identifying header fields.
            line()
        code = field()
        line()
        info = {"code": code, "record": field().split(), "application": field(),
                "date": field().split(), "subject": subject or fname.stem}
        seek("Exercises\t{")
        line()
        times = []
        while (entry := line()).strip() != "}":
            fields = entry.split("\t")
            if len(fields) < 4:
                raise ValueError(f"{fname.name}: incomplete exercise timing")
            try:
                times.append((Fraction(fields[2]), Fraction(fields[3])))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"{fname.name}: invalid exercise timing") from exc
        line()
        rate = field()
        hz = float(rate)
        if not np.isfinite(hz) or hz <= 0 or not times:
            raise ValueError(f"{fname.name}: invalid exercise clock or empty exercise list")
        frame_rate = Fraction(rate)
        seek("Platform {")
        platform = {"ncols": int(field()), "nrows": int(field()), "cell_width": float(field()) / 10,
                    "cell_height": float(field()) / 10, "units": field(), "nframes": int(field()),
                    "freq": float(field())}
        if platform["freq"] != hz or min(platform["nrows"], platform["ncols"]) <= 0:
            raise ValueError(f"{fname.name}: inconsistent platform clock or grid")
        info.update(freq=hz, platform=platform)

        def trials():
            rows, cols = platform["nrows"], platform["ncols"]
            frame_offsets = []
            cursor = 0
            for number, (begin, duration) in enumerate(times, 1):
                # Round both timestamps on the same clock so shared boundaries agree.
                start = round(begin * frame_rate)
                stop = round((begin + duration) * frame_rate)
                if begin < 0 or not 0 <= start < stop <= platform["nframes"]:
                    raise ValueError(f"{fname.name}: invalid frame interval for exercise {number}")
                length = stop - start
                data = np.empty((rows, cols, length))
                if start < cursor:
                    stream.seek(frame_offsets[start])
                    cursor = start
                for frame in range(cursor, stop):
                    if frame == len(frame_offsets):
                        frame_offsets.append(stream.tell())
                    if frame:
                        line()  # separator after previous grid
                    for _ in range(4):
                        line()
                    for row in range(rows):
                        values = line()
                        if frame >= start:
                            samples = values.split("\t")[1:]
                            if len(samples) != cols:
                                raise ValueError(f"{fname.name}: wrong row width in frame {frame}")
                            data[row, :, frame - start] = samples
                cursor = stop
                if not np.isfinite(data).all() or np.any(data < 0):
                    raise ValueError(f"{fname.name}: invalid pressure in exercise {number}")
                grid, warnings, offset = _clean_ascii_trial(data, number)
                timing = {"start_frame": start, "n_frames": length,
                          "first_retained_frame": start + int(offset) if len(grid) else None}
                yield grid, warnings, timing

        yield info, trials()


def read_zebris_text(fname, subject=None, save=True, outdir=None, *, overwrite=False):
    """Read gait pressure ASCII. By default, save frame-first trial NPZs beside the source.

    The archives are named {stem}{trial:03d}.npz in outdir, which is created when
    missing. Existing archives raise FileExistsError unless overwrite is true; the
    check runs before any archive is written.
    """
    fname = Path(fname)
    outdir = Path(outdir) if outdir else fname.parent
    trials, log = [], []
    with open_zebris_trials(fname, subject) as (info, records):
        for data, warnings, _timing in records:
            trials.append(data)
            log.extend(warnings)
    if save:
        targets = [outdir / f"{fname.stem}{number:03d}.npz" for number in range(1, len(trials) + 1)]
        existing = [path.name for path in targets if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {', '.join(existing)} in {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        for path, data in zip(targets, trials, strict=True):
            np.savez_compressed(
                path,
                data=data,
                array_axes=["frame", "y", "x"],
                sampling_rate_hz=info["freq"],
                pressure_units=info["platform"]["units"],
            )
    return info, trials, log
