"""Seekable HTTP reads in a Pyodide worker, including DuckDB's Python filesystem."""

from datetime import UTC, datetime
from functools import lru_cache
from io import BytesIO
from urllib.parse import quote, unquote

from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile

BLOCK_SIZE = 512 * 1024


def _fetch(url, start=None, stop=None):
    # The marimo kernel runs in a Web Worker. A synchronous request here gives
    # DuckDB and zipfile their normal seek/read interface without blocking the UI.
    from js import Uint8Array, XMLHttpRequest

    request = XMLHttpRequest.new()
    request.open("GET", url, False)
    request.responseType = "arraybuffer"
    if start is not None:
        request.setRequestHeader("Range", f"bytes={start}-{stop - 1}")
    request.send()
    if request.status != (200 if start is None else 206):
        raise OSError(f"HTTP {request.status} reading {url}")
    data = Uint8Array.new(request.response).to_py().tobytes()
    if start is not None and len(data) != stop - start:
        raise OSError(f"Incomplete byte range from {url}")
    return data


@lru_cache(maxsize=48)
def _block(url, start, size):
    return _fetch(url, start, min(start + BLOCK_SIZE, size))


class RangeFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        # Share aligned blocks across short-lived DuckDB connections. At most
        # 24 MiB of release bytes are retained after readers close.
        first = start // BLOCK_SIZE * BLOCK_SIZE
        blocks = b"".join(_block(self.path, offset, self.size)
                          for offset in range(first, end, BLOCK_SIZE))
        return blocks[start - first:end - first]


def open_http(url, size):
    if size is None:
        return BytesIO(_fetch(url))
    return RangeFile(None, url, size=size, cache_type="none")


class ReleaseFileSystem(AbstractFileSystem):
    """Make named remote Parquet resources visible to DuckDB without httpfs."""

    protocol = "marimorelease"
    cachable = False

    @classmethod
    def resource(cls, url, size):
        return f"{cls.protocol}://{size or 0}/{quote(url, safe='')}"

    def info(self, path, **kwargs):
        name = self._strip_protocol(path)
        size, url = name.split("/", 1)
        length = int(size) or len(_fetch(unquote(url)))
        return {"name": name, "size": length, "type": "file"}

    def modified(self, path):
        # The pinned resources are immutable.
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _open(self, path, mode="rb", **kwargs):
        if mode != "rb":
            raise ValueError("Release files are read-only")
        size, url = self._strip_protocol(path).split("/", 1)
        return open_http(unquote(url), int(size) or None)
