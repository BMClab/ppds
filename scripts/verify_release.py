"""Verify the files and cross-check every pressure frame, gait contact, pressure map and footprint link."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from reader_data import Release

from ppds import dataset
from ppds.maps import STRIKE_ONSET_N
from ppds.paths import local_data_dir
from ppds.release_layout import (
    PRIVATE_REPORT_PATHS,
    REFERENCE,
    SUBJECT_COLUMNS,
    TABLES,
    VERSION,
    scan_member_count,
    validate_inventory,
    validate_table_schema,
)

EXPECTED_COVERAGE = (95, 3312, 95, 883555, 3312, 3312)
LAST_ROW = dataset.NROWS - 1
# The expected sizes, checksums and row counts come from the dictionary shipped with the
# toolkit, not from the download, so edited files with a matching edited manifest fail.
EXPECTED_METADATA = REFERENCE


def verify_scans(path, resource):
    with zipfile.ZipFile(path) as archive:
        if scan_member_count(archive.namelist()) != resource["members"]:
            raise ValueError("Member count differs from the scan archive manifest")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"CRC mismatch in scan archive for {corrupt}")


def pressure_moments(con, reading_count):
    """Aggregate the physical grid in trial batches so duplicate checks use bounded memory."""
    first, last = con.execute(
        "SELECT min(trial_id), max(trial_id) FROM trials"
    ).fetchone()
    for lower in range(first, last + 1, 64):
        command = (
            "CREATE TEMP TABLE moments AS" if lower == first else "INSERT INTO moments"
        )
        con.execute(
            f"""
            {command}
            SELECT trial_id, frame,
                   CASE WHEN grouping(side) = 1 THEN 'both' ELSE side END AS support,
                   count(*) AS cells, list_unique(list(y::INTEGER * 64 + x)) AS unique_cells,
                   count(*) FILTER (WHERE y BETWEEN 0 AND {LAST_ROW} AND x BETWEEN 0 AND 63
                                      AND value BETWEEN 1 AND 65535 AND side IN ('L', 'R')) AS valid_cells,
                   sum(value)::DOUBLE AS load, sum(x::BIGINT * value)::DOUBLE AS mx,
                   sum(y::BIGINT * value)::DOUBLE AS my, max(value) AS peak,
                   min(y) AS ymin, max(y) AS ymax
            FROM pressure WHERE trial_id BETWEEN ? AND ?
            GROUP BY GROUPING SETS ((trial_id, frame), (trial_id, frame, side))
        """,
            [lower, min(lower + 63, last)],
        )
    # The total catches null and out-of-range trial IDs omitted by the batched scan.
    observed = con.execute(
        "SELECT coalesce(sum(cells), 0) FROM moments WHERE support = 'both'"
    ).fetchone()[0]
    invalid = con.execute("""
        WITH samples AS (
            SELECT trial_id, sum(cells) AS n, min(frame) AS first_frame, max(frame) AS last_frame,
                   bool_and(frame IS NOT NULL AND cells = valid_cells AND cells = unique_cells) AS valid
            FROM moments WHERE support = 'both' GROUP BY trial_id
        )
        SELECT count(*) FROM samples s FULL JOIN trials t USING (trial_id)
        WHERE t.trial_id IS NULL OR coalesce(s.n, 0) IS DISTINCT FROM t.n_cells
           OR NOT s.valid OR s.first_frame < 0 OR s.last_frame >= t.n_frames
    """).fetchone()[0]
    if invalid or observed != reading_count:
        raise ValueError("Inconsistent pressure sensor readings or trial totals")


def verify_trials(con):
    invalid = con.execute("""
        SELECT count(*) FROM trials
        WHERE subject IS NULL OR NOT regexp_full_match(subject, 's[0-9]{3}')
           OR subject NOT BETWEEN 's001' AND 's095'
           OR trial IS NULL OR trial NOT BETWEEN 1 AND 10 OR (condition = 'ss' AND trial != 1)
           OR duration_s IS DISTINCT FROM (n_frames / frequency_hz)
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"Unexpected participant ID, trial number or duration in {invalid} trials"
        )

    counts = con.execute("""
        WITH observed AS (
            SELECT subject,
                count(*) FILTER (WHERE condition = 'ws') AS ws,
                count(*) FILTER (WHERE condition = 'wc') AS wc,
                count(*) FILTER (WHERE condition = 'wf') AS wf,
                count(*) FILTER (WHERE condition = 'rc') AS rc,
                count(*) FILTER (WHERE condition = 'ss') AS ss
            FROM trials GROUP BY subject
        )
        SELECT count(*) FROM subjects JOIN observed USING (subject)
        WHERE TrialsWS IS DISTINCT FROM ws OR TrialsWC IS DISTINCT FROM wc
           OR TrialsWF IS DISTINCT FROM wf OR TrialsRC IS DISTINCT FROM rc
           OR TrialsStanding IS DISTINCT FROM ss
    """).fetchone()[0]
    if counts:
        raise ValueError(
            f"Trial counts differ from reported availability for {counts} participants"
        )


def verify_frames(con):
    invalid = con.execute("""
        WITH frames AS (
            SELECT trial_id, count(*) AS n, count(DISTINCT frame) AS unique_frames,
                   min(frame) AS first_frame, max(frame) AS last_frame
            FROM pressure_features GROUP BY trial_id
        )
        SELECT count(*) FROM frames FULL JOIN trials USING (trial_id)
        WHERE n_frames IS NULL OR n IS NULL OR n != n_frames OR n != unique_frames
           OR first_frame != 0 OR last_frame != n_frames - 1
           OR frequency_hz IS DISTINCT FROM CASE WHEN condition = 'ss' THEN 10 ELSE 100 END
           OR condition IS NULL OR condition NOT IN ('ws', 'wc', 'wf', 'rc', 'ss')
           OR (condition = 'ss' AND n != 100)
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"Inconsistent frame series for {invalid} trials")
    for support, suffix in (("both", ""), ("L", "_left"), ("R", "_right")):
        invalid = con.execute(
            f"""
            SELECT count(*) FROM pressure_features f LEFT JOIN moments m
              ON f.trial_id = m.trial_id AND f.frame = m.frame AND m.support = ?
            WHERE NOT agrees(f.grf_z{suffix}, coalesce(m.load, 0) * (8.469 ^ 2 / 1000))
               OR NOT agrees(f.cop_x{suffix}, 8.469 * m.mx / m.load)
               OR NOT agrees(f.cop_y{suffix}, 8.469 * m.my / m.load)
               OR NOT agrees(f.contact_area{suffix}, coalesce(m.cells, 0) * (8.469 ^ 2 / 100))
               OR f.peak_pressure{suffix} IS DISTINCT FROM coalesce(m.peak, 0)
        """,
            [support],
        ).fetchone()[0]
        if invalid:
            raise ValueError(
                f"Pressure feature mismatch for {support} in {invalid} frames"
            )


def verify_contacts(con):
    invalid = con.execute("""
        WITH ordered AS (
            SELECT *, row_number() OVER w AS ordinal, lag(start_frame) OVER w AS previous_start,
                   lag(stop_frame) OVER (PARTITION BY trial_id, side ORDER BY start_frame, contact)
                       AS previous_side_stop
            FROM contacts
            WINDOW w AS (PARTITION BY trial_id ORDER BY start_frame, contact)
        )
        SELECT count(*) FROM ordered c FULL JOIN trials t USING (trial_id)
        WHERE t.trial_id IS NULL OR (t.condition = 'ss' AND c.ordinal IS NOT NULL)
           OR (t.condition != 'ss' AND c.contact IS NULL)
           OR c.contact IS DISTINCT FROM c.ordinal OR c.side NOT IN ('L', 'R')
           OR (c.ordinal IS NOT NULL AND (c.side IS NULL OR c.start_frame IS NULL OR c.stop_frame IS NULL))
           OR c.start_frame < 0 OR c.stop_frame > t.n_frames OR c.start_frame >= c.stop_frame
           OR c.start_frame <= c.previous_start OR c.start_frame < c.previous_side_stop
           OR (t.condition = 'rc' AND c.start_frame = c.previous_side_stop)
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"Inconsistent gait contact numbering, intervals or trial links in {invalid} rows"
        )
    # Each loaded side/frame contributes all its cell pressures to its gait contact.
    # The null group detects any load outside all contact intervals.
    invalid = con.execute("""
        WITH measured AS (
            SELECT m.trial_id, c.contact, min(m.frame) AS first_frame, max(m.frame)::INTEGER + 1 AS stop,
                   count(*) AS loaded_frames,
                   max(m.peak) AS peak, max(m.load) * (8.469 ^ 2 / 1000) AS force,
                   sum(m.load) * (8.469 ^ 2 / 1000) / t.frequency_hz AS impulse,
                   max(m.cells) * (8.469 ^ 2 / 100) AS area,
                   min(m.ymin) * 8.469 AS ymin, max(m.ymax) * 8.469 AS ymax
            FROM moments m JOIN trials t USING (trial_id) LEFT JOIN contacts c
              ON c.trial_id = m.trial_id AND c.side = m.support
             AND c.start_frame <= m.frame AND m.frame < c.stop_frame
            WHERE t.condition != 'ss' AND m.support != 'both'
            GROUP BY m.trial_id, c.contact, t.frequency_hz
        )
        SELECT count(*) FROM measured m FULL JOIN contacts c USING (trial_id, contact)
        JOIN trials t ON t.trial_id = coalesce(m.trial_id, c.trial_id)
        WHERE m.contact IS NULL OR c.contact IS NULL
           OR c.start_frame IS DISTINCT FROM m.first_frame OR c.stop_frame IS DISTINCT FROM m.stop
           OR (t.condition = 'rc' AND m.loaded_frames != m.stop - m.first_frame)
           OR NOT agrees(c.contact_time, (m.stop - m.first_frame) / t.frequency_hz)
           OR c.peak_pressure IS DISTINCT FROM m.peak OR NOT agrees(c.peak_grf_z, m.force)
           OR NOT agrees(c.vertical_impulse, m.impulse) OR NOT agrees(c.max_contact_area, m.area)
           OR NOT agrees(c.contact_y_min, m.ymin) OR NOT agrees(c.contact_y_max, m.ymax)
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"Gait contact pressure, interval or feature mismatch in {invalid} groups"
        )


def verify_footprints(con):
    invalid = con.execute(f"""
        WITH walking AS (
            SELECT t.subject, t.condition, c.side, c.trial_id, c.contact, c.peak_pressure,
                   row_number() OVER (PARTITION BY t.subject, t.condition, c.side
                                      ORDER BY t.trial, c.contact) - 1 AS footprint
            FROM contacts c JOIN trials t USING (trial_id) WHERE t.condition IN ('ws', 'wc', 'wf')
        ), images AS (
            SELECT subject, condition, side, footprint, min(trial_id) AS trial_id, min(contact) AS contact,
                   max(value) AS image_peak,
                   count(*) = count(*) FILTER (WHERE trial_id IS NOT NULL AND contact IS NOT NULL
                                               AND y BETWEEN 0 AND {LAST_ROW} AND x BETWEEN 0 AND 63
                                               AND isfinite(value) AND value > 0) AS valid,
                   count(*) = list_unique(list(y::INTEGER * 64 + x)) AS unique_pixels,
                   min(trial_id) = max(trial_id) AND min(contact) = max(contact) AS single_event
            FROM footprints GROUP BY subject, condition, side, footprint
        )
        SELECT count(*) FROM walking w FULL JOIN images i USING (subject, condition, side, footprint)
        WHERE w.contact IS NULL OR i.contact IS NULL OR w.trial_id IS DISTINCT FROM i.trial_id
           OR i.image_peak > w.peak_pressure + 1e-8
           OR w.contact IS DISTINCT FROM i.contact OR NOT i.valid OR NOT i.unique_pixels OR NOT i.single_event
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"Inconsistent walking footprint pixels or contact links in {invalid} groups"
        )


def verify_strike_index(con):
    """Recompute each contact's strike index from the frame moments of its foot."""
    invalid = con.execute(f"""
        WITH frames AS (
            SELECT c.trial_id, c.contact, m.frame, m.load, m.my, m.ymin, m.ymax
            FROM contacts c JOIN moments m
              ON m.trial_id = c.trial_id AND m.support = c.side
             AND m.frame >= c.start_frame AND m.frame < c.stop_frame
        ), extent AS (
            SELECT trial_id, contact, min(ymin) AS ymin, max(ymax) AS ymax,
                   arg_min(my / load, frame) FILTER (WHERE load * (8.469 ^ 2 / 1000) >= {STRIKE_ONSET_N}) AS cop_y
            FROM frames GROUP BY trial_id, contact
        )
        SELECT count(*) FROM contacts c LEFT JOIN extent e USING (trial_id, contact)
        WHERE e.cop_y IS NULL OR NOT isfinite(c.strike_index)
           OR NOT agrees(c.strike_index, (e.cop_y - (e.ymin - 0.5)) / (e.ymax - e.ymin + 1))
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"Strike index mismatch in {invalid} gait contacts")


def verify_maps(con):
    """Recompute every map cell from the sensor readings and tie each map to its contact or trial."""
    invalid = con.execute("""
        WITH expected AS (
            SELECT c.trial_id, c.contact, p.y, p.x, max(p.value) AS peak,
                   sum(p.value::INTEGER)::DOUBLE / t.frequency_hz AS pti
            FROM contacts c JOIN pressure p
              ON p.trial_id = c.trial_id AND p.side = c.side
             AND p.frame >= c.start_frame AND p.frame < c.stop_frame
            JOIN trials t ON t.trial_id = c.trial_id
            GROUP BY c.trial_id, c.contact, p.y, p.x, t.frequency_hz
        )
        SELECT count(*) FROM expected e FULL JOIN contact_maps m USING (trial_id, contact, y, x)
        WHERE e.peak IS NULL OR m.peak_pressure IS DISTINCT FROM e.peak
           OR NOT agrees(m.pressure_time_integral, e.pti)
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"Contact map mismatch in {invalid} sensor cells")
    # The integral over the footprint is the contact's vertical impulse, and its peak the contact peak.
    invalid = con.execute("""
        WITH totals AS (
            SELECT trial_id, contact, max(peak_pressure) AS peak,
                   sum(pressure_time_integral) * (8.469 ^ 2 / 1000) AS impulse
            FROM contact_maps GROUP BY trial_id, contact
        )
        SELECT count(*) FROM contacts c FULL JOIN totals m USING (trial_id, contact)
        WHERE m.peak IS DISTINCT FROM c.peak_pressure
           OR abs(m.impulse - c.vertical_impulse) > 1e-9 * greatest(1, abs(c.vertical_impulse))
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"Contact maps disagree with {invalid} gait contact summaries")
    invalid = con.execute("""
        WITH expected AS (
            SELECT p.trial_id, p.side, p.y, p.x, max(p.value) AS peak,
                   sum(p.value::INTEGER)::DOUBLE / t.frequency_hz AS pti
            FROM pressure p JOIN trials t USING (trial_id) WHERE t.condition = 'ss'
            GROUP BY p.trial_id, p.side, p.y, p.x, t.frequency_hz
        ), keyed AS (
            SELECT *, count(*) OVER (PARTITION BY trial_id, y, x) AS copies FROM standing_maps
        )
        SELECT count(*) FROM expected e FULL JOIN keyed m USING (trial_id, side, y, x)
        WHERE e.peak IS NULL OR m.copies IS DISTINCT FROM 1 OR m.peak_pressure IS DISTINCT FROM e.peak
           OR NOT agrees(m.pressure_time_integral, e.pti)
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"Standing map mismatch in {invalid} sensor cells")


def verify(data):
    data = data.resolve()
    source = Release.local(data)
    manifest = source.metadata
    if EXPECTED_METADATA is not None and manifest != EXPECTED_METADATA:
        raise ValueError(
            f"metadata.json differs from the PPDS {VERSION} dictionary distributed with the toolkit"
        )
    resources = manifest["resources"]
    validate_inventory(resources)
    if manifest.get("subject_columns") != list(SUBJECT_COLUMNS):
        raise ValueError(
            "subject_columns differs from the reviewed public participant projection"
        )
    for relative in PRIVATE_REPORT_PATHS:
        if (data / relative).exists():
            raise ValueError(
                f"Excluded file in the release directory: {relative}"
            )
    for name, resource in resources.items():
        path = Path(source.resource_location(name))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != resource["bytes"] or digest != resource["sha256"]:
            raise ValueError(f"Size or checksum mismatch for {resource['path']}")
        if name == "scans":
            verify_scans(path, resource)
        else:
            schema = pq.read_schema(path)
            validate_table_schema(name, schema)
            columns = resource.get("columns")
            if not isinstance(columns, dict) or {
                field.name: str(field.type) for field in schema
            } != {key: spec.get("type") for key, spec in columns.items()}:
                raise ValueError(
                    f"Column dictionary does not match the physical {name} table"
                )
    # The reader's convenience views join sensor and feature rows to trials.
    # Read the physical tables here so an unknown trial ID remains visible.
    with source.connect(TABLES) as con:
        for name in TABLES:
            (count,) = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()
            if count != resources[name]["rows"]:
                raise ValueError(f"Row count mismatch in {name}")
        dataset._subject_view(con, manifest["subject_columns"])
        coverage = con.execute("""
            SELECT (SELECT count(*) FROM subjects), count(*), count(*) FILTER (WHERE condition = 'ss'),
                   sum(n_frames), count(DISTINCT trial_id), count(DISTINCT (subject, condition, trial))
            FROM trials
        """).fetchone()
        if coverage != EXPECTED_COVERAGE:
            raise ValueError(f"Unexpected release coverage: {coverage}")
        verify_trials(con)
        con.execute("SET threads = 4")
        con.execute("""
            CREATE TEMP MACRO agrees(actual, expected) AS
                coalesce(abs(actual - expected) <= 1e-8, actual IS NULL AND expected IS NULL)
        """)
        pressure_moments(con, resources["pressure"]["rows"])
        verify_frames(con)
        verify_contacts(con)
        verify_strike_index(con)
        verify_maps(con)
        verify_footprints(con)
        gait_trials, walking, running = con.execute("""
            SELECT count(DISTINCT trial_id), count(*) FILTER (WHERE condition IN ('ws', 'wc', 'wf')),
                   count(*) FILTER (WHERE condition = 'rc')
            FROM contacts JOIN trials USING (trial_id)
        """).fetchone()
        sides = con.execute("""
            SELECT t.condition, count(*) AS frames, count(*) FILTER (WHERE grf_z_left IS NULL) AS unresolved
            FROM pressure_features f JOIN trials t USING(trial_id) GROUP BY t.condition ORDER BY t.condition
        """).fetchall()
    side_coverage = {
        condition: {"frames": frames, "unresolved_per_foot_frames": unresolved}
        for condition, frames, unresolved in sides
    }
    if manifest.get("pressure_features", {}).get("side_coverage") != side_coverage:
        raise ValueError(
            "Manifest side_coverage differs from observed pressure feature frames"
        )
    return {
        "verified": True,
        "participants": coverage[0],
        "trials": coverage[1],
        "frames": coverage[3],
        "loaded_sensor_readings": resources["pressure"]["rows"],
        "gait_trials": gait_trials,
        "gait_contacts": walking + running,
        "registered_walking_contacts": walking,
        "running_contacts": running,
        "contact_map_cells": resources["contact_maps"]["rows"],
        "standing_map_cells": resources["standing_maps"]["rows"],
        "unresolved_side_frames": {
            condition: unknown for condition, _, unknown in sides
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=local_data_dir(),
                        help="Release directory (default: PPDS_DATA, then data/).")
    args = parser.parse_args(argv)
    data = args.data.resolve()
    print(json.dumps({"data_dir": str(data), **verify(data)}, indent=2))


if __name__ == "__main__":
    main()
