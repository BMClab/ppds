"""File inventory and table keys for a complete PPDS release."""

import json
from importlib.resources import files
from pathlib import PurePosixPath, PureWindowsPath

import pyarrow as pa

TABLES = {
    "trials": (
        "trials.parquet",
        ["trial_id"],
        "Participant characteristics and pressure trial index",
    ),
    "pressure": (
        "pressure.parquet",
        ["trial_id", "frame", "y", "x"],
        "Nonzero plantar pressure readings with anatomical foot side",
    ),
    "pressure_features": (
        "features/pressure_features.parquet",
        ["trial_id", "frame"],
        "Vertical force, center of pressure, loaded area and peak pressure per frame and foot",
    ),
    "contacts": (
        "features/contacts.parquet",
        ["trial_id", "contact"],
        "Walking and running contacts on the original pressure grid",
    ),
    "contact_maps": (
        "features/contact_maps.parquet",
        ["trial_id", "contact", "y", "x"],
        "Peak pressure and pressure-time integral at each sensor during each gait contact",
    ),
    "standing_maps": (
        "features/standing_maps.parquet",
        ["trial_id", "y", "x"],
        "Peak pressure and pressure-time integral at each sensor during quiet standing",
    ),
    "footprints": (
        "features/footprints.parquet",
        ["subject", "condition", "side", "footprint", "y", "x"],
        "Registered walking peak-pressure images linked to their original gait contacts",
    ),
}

# Validate downloaded schemas against the dictionary distributed with the toolkit.
VERSION = "1.1.0"
REFERENCE = json.loads(files("ppds").joinpath("release_1_1_0.json").read_text(encoding="utf-8"))
SUBJECT_COLUMNS = tuple(REFERENCE["subject_columns"])
TABLE_SCHEMAS = {
    name: pa.schema([
        (column, pa.type_for_alias(details["type"]))
        for column, details in REFERENCE["resources"][name]["columns"].items()
    ])
    for name in TABLES
}

PRIVATE_REPORT_PATHS = frozenset(
    relative
    for name in ("spatiotemporal", "grf", "grf_cycle", "standing")
    for relative in (
        f"{name}.parquet",
        f"features/{name}.parquet",
        f"features_{name}.parquet",
    )
) | {"stps_coding.txt"}

SCAN_MEMBERS = frozenset(
    f"scans/s{number:03d}{side}_foot.stl" for number in range(1, 96) for side in "lr"
)


def validate_table_schema(name, schema):
    expected = TABLE_SCHEMAS[name]
    if schema.names != expected.names or schema.types != expected.types:
        raise ValueError(f"Unexpected public {name} schema: {schema}")


def scan_member_count(names):
    if len(names) != len(SCAN_MEMBERS) or set(names) != SCAN_MEMBERS:
        raise ValueError("Incomplete or unexpected members in scan archive")
    return len(names)


def validate_inventory(resources):
    """Require the complete set of named resources and their expected formats."""
    expected = dict.fromkeys(TABLES, "parquet") | {"scans": "zip"}
    actual = {name: spec.get("format") for name, spec in resources.items()}
    if expected != actual:
        differences = sorted(
            name
            for name in expected.keys() | actual.keys()
            if name not in expected or actual.get(name) != expected[name]
        )
        raise ValueError(f"Invalid PPDS resource inventory: {', '.join(differences)}")
    paths = [PurePosixPath(spec["path"]) for spec in resources.values()]
    if any(
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.name == "metadata.json"
        or "\\" in str(path)
        or PureWindowsPath(path).drive
        for path in paths
    ):
        raise ValueError("Invalid release-relative path in PPDS resource inventory")
    filenames = [
        spec.get("zenodo_filename", spec["path"].replace("/", "_"))
        for spec in resources.values()
    ]
    if len(set(paths)) != len(paths) or len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate file in PPDS resource inventory")
    if PRIVATE_REPORT_PATHS.intersection(filenames) or any(
        path.as_posix() in PRIVATE_REPORT_PATHS for path in paths
    ):
        raise ValueError("Excluded filename in PPDS resource inventory")
    if any(
        not name
        or name in {".", "..", "metadata.json"}
        or PurePosixPath(name).name != name
        or "\\" in name
        or PureWindowsPath(name).drive
        for name in filenames
    ):
        raise ValueError("Expected flat deposit filenames in PPDS resource inventory")
