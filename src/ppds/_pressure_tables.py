"""Arrow schemas and sparse pressure encoding for the conversion notebook."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import NCOLS, NROWS

GRID = (NROWS, NCOLS)
PARQUET = {"compression": "zstd", "compression_level": 9, "use_dictionary": True, "write_statistics": True}
GROUP_SIZE = 131_072


PRESSURE = pa.schema(
    [
        ("trial_id", pa.uint16()),
        ("frame", pa.uint16()),
        ("y", pa.uint16()),
        ("x", pa.uint8()),
        ("value", pa.uint16()),
    ],
    metadata={
        b"pressure_unit": b"kPa",
        b"axes": b"frame,y,x",
        b"grid": b"353,64",
        b"sparse": b"omitted cells within recorded frames are zero",
        b"cell_size_cm": b"0.8469",
        b"frame_time_s": b"frame / trials.frequency_hz",
    },
)


TRIAL = pa.schema(
    [
        ("trial_id", pa.uint16()),
        ("subject", pa.string()),
        ("condition", pa.string()),
        ("trial", pa.uint32()),
        ("n_frames", pa.uint16()),
        ("frequency_hz", pa.float64()),
        ("duration_s", pa.float64()),
        ("n_cells", pa.uint32()),
    ],
    metadata={
        b"trial_id": b"zero-based; sorted by subject, condition, trial",
        b"n_frames": b"all recorded frames including zero load",
        b"duration_s": b"n_frames / frequency_hz",
        b"n_cells": b"number of nonzero sensor samples across all frames",
    },
)


def encode_pressure_trial(trial_id, subject, condition, trial, data, frequency):
    """Encode one frame-first N/cm² trial as sparse integer kPa and its index row."""
    if data.ndim != 3 or data.shape[1:] != GRID or len(data) > 65535 or data.dtype.kind not in "uif":
        raise ValueError(f"{subject}/{condition}/{trial}: invalid pressure grid {data.shape} ({data.dtype})")
    frame, y, x = np.nonzero(data)
    source = data[frame, y, x]
    scaled = np.rint(source.astype(np.float64) * 10)
    if not np.all((scaled > 0) & (scaled <= 65535)) or not np.array_equal(
        (scaled / 10).astype(source.dtype), source
    ):
        raise ValueError(f"{subject}/{condition}/{trial}: values cannot be represented in integer kPa")
    pressure = pa.Table.from_arrays(
        [
            np.full(len(frame), trial_id, np.uint16),
            frame,
            y,
            x,
            scaled.astype(np.uint16),
        ],
        schema=PRESSURE,
    )
    row = {
        "trial_id": trial_id,
        "subject": subject,
        "condition": condition,
        "trial": trial,
        "n_frames": len(data),
        "frequency_hz": frequency,
        "duration_s": len(data) / frequency,
        "n_cells": len(pressure),
    }
    return row, pressure


def _write_pressure(trials, stage, *, schema=PRESSURE):
    rows = []
    pending = schema.empty_table()
    with pq.ParquetWriter(
        stage / "pressure.parquet", schema, **PARQUET
    ) as writer:
        for row, table in trials:
            pending = pa.concat_tables([pending, table])
            while len(pending) >= GROUP_SIZE:
                writer.write_table(
                    pending.slice(0, GROUP_SIZE), row_group_size=GROUP_SIZE
                )
                pending = pending.slice(GROUP_SIZE)
            rows.append(row)
            if len(rows) % 250 == 0:
                print(f"Encoded {len(rows)} trials", flush=True)
        if len(pending):
            writer.write_table(pending, row_group_size=GROUP_SIZE)
    return rows
