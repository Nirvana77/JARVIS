"""OTA firmware for edges that can flash themselves (the watch).

One image per device, ``data/firmware/<device_id>.bin`` — the ``build/*.bin``
that ``idf.py build`` writes, copied in as it is. Nothing to configure: the
version is read out of the image's own ``esp_app_desc_t``, so what the brain
announces is exactly what the edge will report after it has flashed it.

An edge that can take an update says what it runs in ``hello`` (``fw``). If the
staged image differs, the brain announces it with ``event {kind: "ota"}`` after
``ready``, and the edge fetches it with ``GET /firmware`` on the same port (see
``RemoteServer.process_request``). Whether and when to flash is the edge's
call: it knows its battery, and whether this version already failed on it.

What the brain *can* know it does not offer (``refusal``): nothing over a dev
build (``-dirty``), which the watch refuses anyway, and nothing older than what
runs. Versions are ``git describe`` output, so "older" is only decidable when
both carry a numeric tag (``v1.2-5-gabc1234``); bare commit hashes from an
untagged repo cannot be ordered, and an image that merely differs is offered.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from jarvis.remote.protocol import _DEVICE_ID_RE

#: ``esp_image_header_t`` (24 bytes) + the first ``esp_image_segment_header_t``
#: (8 bytes); the app description is the first thing in that segment.
_IMAGE_MAGIC = 0xE9
_APP_DESC_OFFSET = 32
_APP_DESC_MAGIC = 0xABCD5432
#: magic, secure_version, reserv1[2], version[32], project_name[32]
_APP_DESC = struct.Struct("<II8s32s32s")


#: ``git describe --tags --dirty``: ``<tag>[-<n>-g<hash>][-dirty]``, the tag
#: numeric (``v1``, ``1.2``, ``v1.2.3``). A bare all-digit hash is not a tag.
_DESCRIBE_RE = re.compile(r"^v?(?P<tag>\d+(?:\.\d+)*)(?:-(?P<n>\d+)-g[0-9a-f]+)?$")


def _order(version: str) -> tuple | None:
    """A sort key for a described version, or ``None`` if it has none."""
    m = _DESCRIBE_RE.match(version)
    if m is None or (not version.startswith("v") and "." not in m["tag"]):
        return None
    return tuple(int(part) for part in m["tag"].split(".")), int(m["n"] or 0)


def refusal(running: str, staged: str) -> str | None:
    """Why an edge running ``running`` should not be offered ``staged``:
    ``current`` (it runs it), ``dev`` (a ``-dirty`` build, which the edge keeps),
    ``newer`` (it runs a later version) — or ``None`` to offer it."""
    if staged == running:
        return "current"
    if running.endswith("-dirty"):
        return "dev"
    have, offered = _order(running), _order(staged)
    if have is not None and offered is not None and have > offered:
        return "newer"
    return None


@dataclass(frozen=True)
class Firmware:
    path: Path
    version: str
    project: str
    size: int
    sha256: str

    def announcement(self, path: str) -> dict:
        return {"version": self.version, "size": self.size, "sha256": self.sha256, "path": path}


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def read_image(path: Path) -> Firmware | None:
    """The image's version and digest, or ``None`` if it is not an ESP-IDF app."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    end = _APP_DESC_OFFSET + _APP_DESC.size
    if len(data) < end or data[0] != _IMAGE_MAGIC:
        return None
    magic, _secure, _reserved, version, project = _APP_DESC.unpack_from(data, _APP_DESC_OFFSET)
    if magic != _APP_DESC_MAGIC:
        return None
    return Firmware(
        path=Path(path),
        version=_cstr(version),
        project=_cstr(project),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


class FirmwareStore:
    """The staged images, re-read only when a file changes."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._cache: dict[str, tuple[tuple[int, int], Firmware | None]] = {}

    def get(self, device_id: str) -> Firmware | None:
        # The id becomes a filename; hello already checked it, the HTTP header
        # has not been.
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.match(device_id):
            return None
        path = self.directory / f"{device_id}.bin"
        try:
            st = path.stat()
        except OSError:
            self._cache.pop(device_id, None)
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        cached = self._cache.get(device_id)
        if cached and cached[0] == stamp:
            return cached[1]
        image = read_image(path)
        self._cache[device_id] = (stamp, image)
        return image

    def offer(self, device_id: str, running: str | None) -> Firmware | None:
        """The image to announce to an edge running ``running``, if any. An
        edge that did not say what it runs cannot flash itself."""
        if not running:
            return None
        image = self.get(device_id)
        if image is None or refusal(running, image.version):
            return None
        return image
