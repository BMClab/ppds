"""Seekable HTTP reads in a Pyodide worker, including DuckDB's Python filesystem."""

from collections import OrderedDict
from datetime import UTC, datetime
from functools import lru_cache
from io import BytesIO
from urllib.parse import quote, unquote

from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile

BLOCK_SIZE = 512 * 1024
# At most 24 MiB shared across short-lived DuckDB connections in this worker.
_blocks = OrderedDict()


def _request(url, start=None, stop=None):
    # The marimo kernel runs in a Web Worker. A synchronous request here gives
    # DuckDB and zipfile their normal seek/read interface without blocking the UI.
    from js import Uint8Array, XMLHttpRequest

    request = XMLHttpRequest.new()
    request.open("GET", url, False)
    request.responseType = "arraybuffer"
    if start is not None:
        request.setRequestHeader("Range", f"bytes={start}-{stop - 1}")
    request.send()
    if request.status != 200 and not (start is not None and request.status == 206):
        raise OSError(f"HTTP {request.status} reading {url}")
    data = Uint8Array.new(request.response).to_py().tobytes()
    if request.status == 206 and len(data) != stop - start:
        raise OSError(f"Incomplete byte range from {url}")
    return request.status, data


def _fetch(url):
    return _request(url)[1]


class RangeFile(AbstractBufferedFile):
    _full_file = None

    def _fetch_range(self, start, end):
        end = min(end, self.size)
        if start >= end:
            return b""
        if self._full_file is not None:
            return self._full_file[start:end]
        first = start // BLOCK_SIZE * BLOCK_SIZE
        blocks = []
        for offset in range(first, end, BLOCK_SIZE):
            key = (self.path, self.size, offset)
            if key in _blocks:
                _blocks.move_to_end(key)
            else:
                status, data = _request(self.path, offset, min(offset + BLOCK_SIZE, self.size))
                if status == 200:
                    # Some mirrors ignore Range. Retain large responses only on this reader.
                    if len(data) != self.size:
                        raise OSError(f"Unexpected full response size from {self.path}")
                    if self.size > BLOCK_SIZE:
                        self._full_file = data
                        return data[start:end]
                _blocks[key] = data
                if len(_blocks) > 48:
                    _blocks.popitem(last=False)
            blocks.append(_blocks[key])
        return b"".join(blocks)[start - first:end - first]

    def close(self):
        self._full_file = None
        super().close()


def open_http(url, size):
    if size is None:
        return BytesIO(_fetch(url))
    return RangeFile(None, url, size=size, cache_type="none")


class ReleaseFileSystem(AbstractFileSystem):
    """Make named remote Parquet resources visible to DuckDB without httpfs."""

    protocol = "marimorelease"
    cachable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # DuckDB can ask for the size before opening an unsized resource.
        self._whole_file = lru_cache(maxsize=1)(_fetch)

    @classmethod
    def resource(cls, url, size):
        length = "unknown" if size is None else str(size)
        return f"{cls.protocol}://{length}/{quote(url, safe='')}"

    def info(self, path, **kwargs):
        name = self._strip_protocol(path)
        size, url = name.split("/", 1)
        length = len(self._whole_file(unquote(url))) if size == "unknown" else int(size)
        return {"name": name, "size": length, "type": "file"}

    def modified(self, path):
        # The pinned resources are immutable.
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _open(self, path, mode="rb", **kwargs):
        if mode != "rb":
            raise ValueError("Release files are read-only")
        size, url = self._strip_protocol(path).split("/", 1)
        if size == "unknown":
            return BytesIO(self._whole_file(unquote(url)))
        return open_http(unquote(url), int(size))
