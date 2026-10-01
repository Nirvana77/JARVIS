"""Pairing: how a device without a token gets one.

The device sends ``pair`` with an X25519 public key; the brain answers
``pairing`` with its own. Both compute the shared secret and from it (bound to
the device id and both public keys) a 32-byte key and a 6-digit code. The
device shows the code. A person who can reach the brain confirms it
(``python -m jarvis pair <code>``, which needs the brain's admin secret): that
approves the device, and a matching code also proves nobody sat in between and
swapped the keys. The brain then sends ``paired``: a fresh token sealed with
AES-256-GCM under the shared key. The device keeps the token (the watch in its
flash) and from then on says ``hello`` like a device configured by hand.

The link is normally TLS already; the sealing is for when it is not (``ws://``
on a LAN), and the code is what makes the exchange worth anything at all.

Derivation, which the watch's C (mbedtls) mirrors byte for byte:

    transcript = b"jarvis-pair-v1\\0" + device_id + b"\\0" + device_pub + brain_pub
    key  = SHA-256(b"key\\0"  + shared + transcript)
    code = big-endian uint32 of SHA-256(b"code\\0" + shared + transcript)[:4] % 1_000_000
    sealed token = AES-256-GCM(key, 12-byte nonce, token, aad = device_id)
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: (key hex, code) for shared = bytes(range(32)), device "watch", device_pub
#: 32 x 0x01, brain_pub 32 x 0x02 — for checking another implementation.
KNOWN_VECTOR = ("56f696b7edf059b5682aa360d01b6e68cdde3d0263ef1a5a7877a9cdcd8c3b3c", "079006")


@dataclass(frozen=True)
class Secret:
    key: bytes   # AES-256-GCM key for the token
    code: str    # 6 digits, shown on the device, confirmed on the brain


def public_key(raw: bytes) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(raw)


def derive(shared: bytes, device_id: str, device_pub: bytes, brain_pub: bytes) -> Secret:
    transcript = b"jarvis-pair-v1\0" + device_id.encode() + b"\0" + device_pub + brain_pub
    key = hashlib.sha256(b"key\0" + shared + transcript).digest()
    digest = hashlib.sha256(b"code\0" + shared + transcript).digest()
    code = int.from_bytes(digest[:4], "big") % 1_000_000
    return Secret(key=key, code=f"{code:06d}")


def seal(key: bytes, device_id: str, token: str) -> tuple[bytes, bytes]:
    nonce = os.urandom(12)
    return nonce, AESGCM(key).encrypt(nonce, token.encode(), device_id.encode())


def open_box(key: bytes, device_id: str, nonce: bytes, box: bytes) -> str:
    return AESGCM(key).decrypt(nonce, box, device_id.encode()).decode()


def _write_private(path: Path, text: str) -> None:
    """Readable by this user only, and never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


class TokenStore:
    """``data/remote/tokens.json``: tokens handed out by pairing, per device.
    Next to the ``.env`` ones (``JARVIS_EDGE_TOKENS``), which still work."""

    def __init__(self, remote_dir: str | Path) -> None:
        self.path = Path(remote_dir) / "tokens.json"

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def get(self, device_id: str) -> str | None:
        return self._read().get(device_id)

    def devices(self) -> list[str]:
        return sorted(self._read())

    def set(self, device_id: str, token: str) -> None:
        data = self._read()
        data[device_id] = token
        _write_private(self.path, json.dumps(data, indent=2))

    def remove(self, device_id: str) -> bool:
        data = self._read()
        if data.pop(device_id, None) is None:
            return False
        _write_private(self.path, json.dumps(data, indent=2))
        return True


def admin_secret(remote_dir: str | Path) -> str:
    """``data/remote/admin.token``, made on first use: what approving a pairing
    takes. Reading it takes this machine's filesystem, which is the point —
    behind a tunnel, a request from 127.0.0.1 proves nothing."""
    path = Path(remote_dir) / "admin.token"
    try:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32:
            return value
    except OSError:
        pass
    value = secrets.token_urlsafe(32)
    _write_private(path, value + "\n")
    return value
