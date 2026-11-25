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
        return 206, payloads[url][start:stop]

    monkeypatch.setattr(_browser, "_request", fetch)
    _browser._blocks.clear()
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
    _browser._blocks.clear()


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


@pytest.mark.parametrize("response", ["full", "partial", "hidden-range", "short-full", "short-partial",
                                      "wrong-range", "missing"])
def test_browser_range_http_responses(response, monkeypatch):
    import sys
    from types import SimpleNamespace

    from reader_data import _browser

    contents = bytes(range(31))
    requests = []
    monkeypatch.setattr(_browser, "BLOCK_SIZE", 8)

    class Request:
        def open(self, method, url, asynchronous):
            assert (method, asynchronous) == ("GET", False)

        def setRequestHeader(self, key, value):
            assert key == "Range"
            start, last = map(int, value.removeprefix("bytes=").split("-"))
            self.part = slice(start, last + 1)

        def send(self):
            requests.append(self.part)
            self.status = {"full": 200, "partial": 206, "hidden-range": 206, "short-full": 200,
                           "short-partial": 206, "wrong-range": 206, "missing": 404}[response]
            data = contents[self.part] if self.status == 206 else contents
            self.response = memoryview(data[:-1] if response.startswith("short") else data)

        def getResponseHeader(self, name):
            assert name == "Content-Range"
            if response == "hidden-range":
                return None
            # A misbehaving cache returns the right length from the wrong offset.
            start = self.part.start + (8 if response == "wrong-range" else 0)
            return f"bytes {start}-{start + self.part.stop - self.part.start - 1}/{len(contents)}"

    js = SimpleNamespace(XMLHttpRequest=SimpleNamespace(new=Request),
                         Uint8Array=SimpleNamespace(new=lambda data: SimpleNamespace(to_py=lambda: data)))
    monkeypatch.setitem(sys.modules, "js", js)
    with _browser.open_http(f"https://example.org/{response}", len(contents)) as stream:
        assert stream._fetch_range(13, 13) == stream._fetch_range(13, 11) == b""
        assert not requests
        stream.seek(10)
        if response in ("full", "partial", "hidden-range"):
            assert stream.read(12) == contents[10:22]
            stream.seek(3)
            assert stream.read(4) == contents[3:7]
            assert len(requests) == (1 if response == "full" else 3)
        else:
            with pytest.raises(OSError):
                stream.read(4)


def test_full_responses_are_shared_by_later_readers(monkeypatch):
    from reader_data import _browser

    contents = bytes(range(40))
    requests = []

    def ignore_range(url, start=None, stop=None):
        requests.append(url)
        return 200, contents

    monkeypatch.setattr(_browser, "BLOCK_SIZE", 8)
    monkeypatch.setattr(_browser, "_request", ignore_range)
    monkeypatch.setattr(_browser, "_full_files", {})
    for _ in range(3):
        with _browser.open_http("https://example.org/no-range", len(contents)) as stream:
            stream.seek(30)
            assert stream.read(5) == contents[30:35]
    with _browser.open_http("https://example.org/other", len(contents)) as stream:
        assert stream.read(3) == contents[:3]
    assert requests == ["https://example.org/no-range", "https://example.org/other"]
    assert list(_browser._full_files) == [("https://example.org/other", len(contents))]
