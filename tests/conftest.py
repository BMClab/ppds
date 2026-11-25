"""Run PPDS release integration tests against local files and a range-capable HTTP mirror."""

import http.server
import re
import threading
from collections import Counter
from contextlib import contextmanager
from urllib.parse import unquote, urlsplit

import pytest
import reader_data

from ppds import dataset
from ppds.paths import local_data_dir


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Static HTTP fixture supporting bytes=a-b, a-, and -n, with per-file counters."""

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self._serve(body=False)

    def do_GET(self):
        self._serve(body=True)

    def _serve(self, body):
        name = unquote(urlsplit(self.path).path).lstrip("/")
        zenodo = re.fullmatch(r"api/records/[0-9]+/files/([^/]+)/content", name)
        if zenodo:
            name = zenodo.group(1)
        path = self.server.files.get(name)
        if path is None:
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            first, last = match.groups()
            start = int(first) if first else max(0, size - int(last))
            end = min(int(last), size - 1) if first and last else size - 1
            if not 0 <= start <= end < size:
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        if body or name not in self.server.chunked_head:
            self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if body:
            remaining = end - start + 1
            with self.server.counter_lock:
                self.server.sent[name] += remaining
                self.server.ranges[name] += int(bool(match))
            with path.open("rb") as stream:
                stream.seek(start)
                while remaining:
                    chunk = stream.read(min(remaining, 256 * 1024))
                    self.wfile.write(chunk)
                    remaining -= len(chunk)


@contextmanager
def range_server(files):
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler) as server:
        server.files = files
        server.chunked_head = set()
        server.sent = Counter()
        server.ranges = Counter()
        server.counter_lock = threading.Lock()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/", server
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture(scope="session")
def data_root():
    root = local_data_dir()
    if not any((root / name).exists() for name in ("metadata.json", "trials.parquet", "pressure.parquet", "scans.zip")):
        pytest.skip("requires a prepared PPDS release (set PPDS_DATA to its directory)")
    return root


@pytest.fixture(scope="session")
def http_root(data_root):
    files = {
        "_".join(path.relative_to(data_root).parts): path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".parquet", ".zip"}
    }
    with range_server(files) as mirror:
        yield mirror


@pytest.fixture(params=["local", "http"])
def source(request, data_root, monkeypatch):
    if request.param == "http":
        url, _ = request.getfixturevalue("http_root")
        monkeypatch.setattr(reader_data, "ZENODO_FILES", url + "api/records/{record}/files/")
        dataset.use_zenodo(1234)
    else:
        dataset.use_local(data_root)
    try:
        yield request.param
    finally:
        dataset.use_local()
