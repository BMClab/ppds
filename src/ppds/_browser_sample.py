"""Load the checksum-pinned pressure preview in the browser's virtual filesystem."""

import hashlib
import io
from functools import lru_cache
from tempfile import TemporaryDirectory
from zipfile import ZipFile

SOURCE = ("https://raw.githubusercontent.com/BMClab/ppds/"
          "fbb2f02d1b83acc82f606afaa64ea8169fe9d07b/notebook-assets/pressure-sample.zip")
SHA256 = "52417208581f04de5cca1f0270fe6ec077cc83045d8944cc7b878ea7c50ca4e5"


@lru_cache(maxsize=1)
def load():
    from reader_data._browser import open_http

    with open_http(SOURCE, None) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("The downloaded PPDS browser sample failed its SHA-256 check")
    temporary = TemporaryDirectory(prefix="ppds-preview-")
    with ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(temporary.name)
    return temporary
