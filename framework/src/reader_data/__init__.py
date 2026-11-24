"""DuckDB access to named Parquet resources in a local or hosted data release."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

import duckdb
import fsspec

ZENODO_FILES = "https://zenodo.org/api/records/{record}/files/"


def _relative_path(path):
    path = str(path)
    relative = PurePosixPath(path)
    if (not relative.parts or relative.is_absolute() or ".." in relative.parts
            or "\\" in path or urlsplit(path).scheme):
        raise ValueError(f"Expected a release-relative path, got {path!r}")
    return relative


@dataclass(frozen=True)
class Release:
    """A release root with a metadata.json file declaring resources by name and path.

    Use local(), http() or zenodo() to select the source. Metadata is fetched
    on first use. connect() returns an independent DuckDB connection.
    """

    root: Path | str
    suffix: str = ""
    flat: bool = False

    @classmethod
    def local(cls, directory):
        """Open a directory containing metadata.json and its release resources."""
        return cls(Path(directory).expanduser().resolve())

    @classmethod
    def http(cls, base, *, suffix="", flat=False):
        """Open an HTTP(S) directory. Flat servers replace path separators with '_'."""
        url = urlsplit(base)
        if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
            raise ValueError("Expected an HTTP(S) base directory URL")
        return cls(base.rstrip("/") + "/", suffix, flat)

    @classmethod
    def zenodo(cls, record):
        """Open the public files in a numbered Zenodo record."""
        if not str(record).isascii() or not str(record).isdecimal():
            raise ValueError("Expected a numeric Zenodo record ID")
        return cls.http(ZENODO_FILES.format(record=record), suffix="/content", flat=True)

    def location(self, path):
        """Resolve a release-relative POSIX file path to a local path or encoded URL."""
        relative = _relative_path(path)
        if isinstance(self.root, Path):
            nested = self.root / relative
            # Downloading all files from a Zenodo record yields a flat directory.
            if len(relative.parts) > 1 and not nested.is_file():
                flat = self.root / "_".join(relative.parts)
                if flat.is_file():
                    return str(flat)
            return str(nested)
        name = relative.as_posix()
        if self.flat:
            name = name.replace("/", "_")
        return self.root + quote(name, safe="/") + self.suffix

    def open_file(self, path, *, size=None):
        """Open a binary stream. Known byte size permits seeking even with a chunked HTTP HEAD."""
        return self._open(self.location(path), size)

    def _open(self, target, size):
        if isinstance(self.root, Path):
            return open(target, "rb")
        http = fsspec.filesystem("http")
        return http.open(target, "rb", block_size=64 * 1024, cache_type="readahead", size=size)

    def open_resource(self, name):
        """Open a named resource, using its manifest size for remote byte ranges."""
        return self._open(self.resource_location(name), self.metadata["resources"][name].get("bytes"))

    @cached_property
    def metadata(self):
        """The decoded metadata.json document."""
        try:
            with self.open_file("metadata.json") as stream:
                return json.load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Release metadata missing at {self.location('metadata.json')}. "
                "Select the directory containing the downloaded metadata.json or configure a hosted release."
            ) from exc

    def resource_path(self, name):
        """Return the named resource's path for the selected source layout."""
        spec = self.metadata["resources"][name]
        path = spec["path"]
        deposit_name = spec.get("zenodo_filename", path.replace("/", "_"))
        if self.flat:
            return deposit_name
        if (isinstance(self.root, Path) and not (self.root / _relative_path(path)).is_file()
                and (self.root / _relative_path(deposit_name)).is_file()):
            return deposit_name
        return path

    def resource_location(self, name):
        """Resolve a named resource using the paths declared in the manifest."""
        path = self.resource_path(name)
        if isinstance(self.root, Path):
            return str(self.root / _relative_path(path))
        return self.location(path)

    def table(self, name, *, connection=None):
        """Return a lazy DuckDB relation, optionally in an existing SQL session."""
        if PurePosixPath(self.metadata["resources"][name]["path"]).suffix.lower() != ".parquet":
            raise ValueError(f"Resource {name!r} is not a Parquet table")
        con = connection if connection is not None else duckdb.connect()
        return con.read_parquet(self.resource_location(name), hive_partitioning=False)

    def connect(self, names=None):
        """Open a SQL session with views for the requested (default: all) Parquet resources.

        Use as a context manager or close the connection after your queries.
        """
        if names is None:
            names = [name for name, spec in self.metadata["resources"].items()
                     if PurePosixPath(spec["path"]).suffix.lower() == ".parquet"]
        elif isinstance(names, str):
            names = [names]
        con = duckdb.connect()
        try:
            for name in names:
                self.table(name, connection=con).create_view(name)
        except BaseException:
            con.close()
            raise
        return con
