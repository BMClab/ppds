# Shared release reader

`reader_data.Release` opens a local data directory, an HTTP mirror or a numbered public Zenodo record. It installs with the PPDS environment through `uv sync`. To use the reader in another local project, install the [framework directory](https://github.com/BMClab/ppds/tree/main/framework) as a Python package.

A release places `metadata.json` at its root. Each key in `resources` is the name of a resource. Its `path` is relative to the release directory. An optional `zenodo_filename` gives the flat filename used on Zenodo. The default replaces each slash with an underscore. A downloaded record can retain those flat filenames.

```json
{
  "resources": {
    "observations": {"path": "tables/observations.parquet"},
    "sites": {"path": "tables/sites.parquet"},
    "meshes": {"path": "meshes.zip"}
  }
}
```

```python
from reader_data import Release

release = Release.local("data")
with release.connect(["sites", "observations"]) as con:
    rows = con.sql("""
        SELECT site, avg(value) AS mean_value
        FROM observations JOIN sites USING (site)
        GROUP BY site ORDER BY site
    """).fetchall()

values = release.table("observations").filter("site = 5").project("value").fetchall()
```

Use `Release.http(base_url)` for a directory on an HTTP(S) server or `Release.zenodo(record_id)` for a published version. Read with the same queries against any of these sources. For a hosted release, DuckDB's HTTP filesystem extension uses byte ranges to fetch the needed Parquet data. Keeping a connection open across queries lets DuckDB reuse its session cache.

`release.metadata` returns the release descriptor. `release.connect()` exposes all Parquet resources as SQL views. Use `open_resource` for seekable binary access. Including a resource’s size in the `bytes` field permits seeking even when an HTTP server omits its content length.

```python
from zipfile import ZipFile

with release.open_resource("meshes") as stream, ZipFile(stream) as archive:
    mesh = archive.read("meshes/sample.stl")
```

For dataset-specific trial, pressure, force and mesh operations, see the [PPDS package guide](../../../src/ppds/README.md).
