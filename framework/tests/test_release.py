"""Tests for a study-independent, named-resource release."""

import json

import duckdb
import pytest
from reader_data import Release


def test_query_named_tables_and_ignore_non_tabular_resources(tmp_path):
    tables = tmp_path / "observations"
    tables.mkdir()
    with duckdb.connect() as con:
        con.sql("SELECT * FROM (VALUES (1, 23), (2, 45)) t(site, value)").to_parquet(str(tables / "today's.parquet"))
        con.sql("SELECT 2 AS site, 'north' AS area").to_parquet(str(tables / "locations.parquet"))
    resources = {"measurements": {"path": "observations/today's.parquet"},
                 "sampling sites": {"path": "observations/locations.parquet"}, "attachment": {"path": "image.zip"}}
    (tmp_path / "metadata.json").write_text(json.dumps({"resources": resources}))
    release = Release.local(tmp_path)
    with release.connect() as con:
        assert con.sql('SELECT value FROM measurements JOIN "sampling sites" USING (site)').fetchall() == [(45,)]
        assert {t[0] for t in con.sql("SHOW TABLES").fetchall()} == {"measurements", "sampling sites"}
    with release.connect("measurements") as con:
        assert con.sql("SHOW TABLES").fetchall() == [("measurements",)]
    assert release.table("measurements").filter("site=2").fetchall() == [(2, 45)]
    with pytest.raises(ValueError, match="not a Parquet table"):
        release.table("attachment")


def test_remote_named_locations():
    assert Release.http("https://example.org/data").location("tables/a b.parquet") == (
        "https://example.org/data/tables/a%20b.parquet")
    assert Release.zenodo(1234).location("tables/a.parquet") == (
        "https://zenodo.org/api/records/1234/files/tables_a.parquet/content")
    with pytest.raises(ValueError, match="record ID"):
        Release.zenodo("abc")


@pytest.mark.parametrize("path", ["../outside", "/absolute", "https://elsewhere.test/file", ""])
def test_release_paths_stay_relative(tmp_path, path):
    with pytest.raises(ValueError, match="release-relative"):
        Release.local(tmp_path).location(path)


def test_read_flat_download_with_declared_deposit_name(tmp_path):
    spec = {"path": "tables/measurements.parquet", "zenodo_filename": "published_measurements.parquet"}
    (tmp_path / "metadata.json").write_text(json.dumps({"resources": {"measurements": spec}}))
    duckdb.sql("SELECT 17 AS k").to_parquet(str(tmp_path / spec["zenodo_filename"]))
    assert Release.local(tmp_path).table("measurements").fetchall() == [(17,)]


def test_missing_resource_preserves_declared_local_path(tmp_path):
    spec = {"path": "features/cells.parquet", "zenodo_filename": "published.parquet"}
    (tmp_path / "metadata.json").write_text(json.dumps({"resources": {"cells": spec}}))
    assert Release.local(tmp_path).resource_path("cells") == "features/cells.parquet"
    # A file with the default flat spelling cannot take an explicitly named resource's place.
    (tmp_path / "features_cells.parquet").touch()
    with pytest.raises(FileNotFoundError):
        Release.local(tmp_path).open_resource("cells")


def test_declared_flat_filename_wins_when_default_name_also_exists(tmp_path):
    spec = {"path": "tables/measurements.parquet", "zenodo_filename": "published.parquet"}
    (tmp_path / "metadata.json").write_text(json.dumps({"resources": {"measurements": spec}}))
    duckdb.sql("SELECT 17 AS k").to_parquet(str(tmp_path / "published.parquet"))
    duckdb.sql("SELECT 25 AS k").to_parquet(str(tmp_path / "tables_measurements.parquet"))
    assert Release.local(tmp_path).table("measurements").fetchall() == [(17,)]


def test_browser_reader_reuses_byte_ranges_and_distinguishes_sources(tmp_path, monkeypatch):
    from reader_data import _browser

    payloads = {}
    for value in (13, 27):
        path = tmp_path / f"{value}.parquet"
        duckdb.sql(f"SELECT {value} AS measurement").to_parquet(str(path))
        payloads[f"https://example.org/{value}/table.parquet"] = path.read_bytes()
    reads = []

    def fetch(url, start=None, stop=None):
        reads.append((url, start, stop))
        return payloads[url][start:stop]

    monkeypatch.setattr(_browser, "_fetch", fetch)
    _browser._block.cache_clear()
    fs = _browser.ReleaseFileSystem()
    with duckdb.connect() as con:
        con.register_filesystem(fs)
        for url, data in payloads.items():
            name = fs.resource(url, len(data))
            assert con.read_parquet(name).fetchall()[0][0] == int(url.split("/")[-2])
            with _browser.open_http(url, len(data)) as stream:
                stream.seek(-4, 2)
                assert stream.read() == b"PAR1"
    assert reads == [(url, 0, len(data)) for url, data in payloads.items()]
    _browser._block.cache_clear()


@pytest.mark.parametrize("contents", [b"hello", b""])
def test_unknown_size_browser_file_reuses_download(contents, monkeypatch):
    from reader_data import _browser

    reads = []

    def fetch(url):
        reads.append(url)
        return contents

    monkeypatch.setattr(_browser, "_fetch", fetch)
    fs = _browser.ReleaseFileSystem()
    url = "https://example.org/unknown"
    path = fs.resource(url, None)
    assert fs.info(path)["size"] == len(contents)
    with fs.open(path) as first, fs.open(path) as second:
        assert first.read() == second.read() == contents
    empty = fs.resource("https://example.org/empty", 0)
    assert fs.size(empty) == 0
    with fs.open(empty) as stream:
        assert stream.read() == b""
    assert reads == [url]
