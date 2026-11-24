"""Load the checksum-pinned pressure preview in the browser's virtual filesystem."""

import hashlib
import io
from functools import lru_cache
from tempfile import TemporaryDirectory
from zipfile import ZipFile

SOURCE = ("https://raw.githubusercontent.com/BMClab/ppds/"
          "84c6c9111934e716177212af883230ea66c89866/pressure-sample.zip")
SHA256 = "50b4387d2867d9dbe9abebc813566e60f76ad4f85aae5b5d5b7617333bd21650"


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
