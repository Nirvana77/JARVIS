"""OTA firmware for edges that can flash themselves (the watch).

The brain keeps one image per device in ``data/firmware/<device_id>.bin``. An
edge that reports its running version in ``hello`` (``fw``) is told about a
different image right after ``ready``, and fetches it with a plain
``GET /firmware`` on the same port, authorised with its own token — so the
tunnel, the TLS and the secret are the ones it already has.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct

import pytest

from jarvis.remote import protocol as P
from jarvis.remote.firmware import FirmwareStore, read_image
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeEdge, make_config


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


async def close(brain):
    brain.ws_server.close()
    await brain.ws_server.wait_closed()


def fake_image(version="1.2.3", project="jarvis_edge", payload=b"\x55" * 4096) -> bytes:
    """Enough of an ESP-IDF app image for the brain: the image header, the
    first segment header, then ``esp_app_desc_t``."""
    header = bytes([0xE9]) + bytes(23)
    segment = bytes(8)
    desc = struct.pack(
        "<III", 0xABCD5432, 0, 0
    ) + bytes(4)  # magic, secure_version, reserv1[2]
    desc += version.encode().ljust(32, b"\0")
    desc += project.encode().ljust(32, b"\0")
    desc += b"12:00:00".ljust(16, b"\0") + b"Sep 30 2026".ljust(16, b"\0")
    desc += b"v5.5.5".ljust(32, b"\0") + bytes(32)
    return header + segment + desc + payload


def put_image(config, data: bytes, device=DEVICE):
    config.firmware_dir.mkdir(parents=True, exist_ok=True)
    (config.firmware_dir / f"{device}.bin").write_bytes(data)


async def http_get(port, path="/firmware", headers=None) -> tuple[int, dict, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    lines = [f"GET {path} HTTP/1.1", f"Host: 127.0.0.1:{port}"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line, *header_lines = head.decode().split("\r\n")
    got = {}
    for line in header_lines:
        key, _, value = line.partition(":")
        got[key.strip().lower()] = value.strip()
    return int(status_line.split()[1]), got, body


def auth(token=TOKEN, device=DEVICE) -> dict:
    return {"Authorization": f"Bearer {token}", "X-Jarvis-Device": device}


# -- reading an image ------------------------------------------------------


def test_the_version_is_read_from_the_image_itself(tmp_path):
    data = fake_image("v0.3-2-gabc123")
    path = tmp_path / "x.bin"
    path.write_bytes(data)
    image = read_image(path)
    assert image.version == "v0.3-2-gabc123"
    assert image.project == "jarvis_edge"
    assert image.size == len(data)
    assert image.sha256 == hashlib.sha256(data).hexdigest()


def test_a_file_that_is_not_an_app_image_is_ignored(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"not firmware at all" * 10)
    assert read_image(path) is None
    (tmp_path / "short.bin").write_bytes(b"\xe9")
    assert read_image(tmp_path / "short.bin") is None


# -- what is offered -------------------------------------------------------


def test_an_image_is_offered_only_when_it_differs_from_what_runs(tmp_path):
    config = make_config(tmp_path)
    store = FirmwareStore(config.firmware_dir)
    assert store.offer(DEVICE, "1.0") is None  # nothing staged
    put_image(config, fake_image("1.1"))
    assert store.offer(DEVICE, "1.1") is None  # already running it
    assert store.offer(DEVICE, None) is None  # an edge that cannot flash itself
    assert store.offer(DEVICE, "1.0").version == "1.1"
    assert store.offer("kitchen", "1.0") is None  # another device's image


def test_a_dev_build_is_left_alone(tmp_path):
    # -dirty was flashed over USB on purpose, and the watch refuses anything
    # over it (ota.c) — so the brain does not pretend to offer it.
    config = make_config(tmp_path)
    store = FirmwareStore(config.firmware_dir)
    put_image(config, fake_image("v1.0-3-gbbbbbbb"))
    assert store.offer(DEVICE, "v1.0-5-gaaaaaaa-dirty") is None
    assert store.offer(DEVICE, "f39716b-dirty") is None


def test_an_older_image_is_not_offered_when_the_versions_can_be_ordered(tmp_path):
    config = make_config(tmp_path)
    store = FirmwareStore(config.firmware_dir)
    put_image(config, fake_image("v1.0-3-gbbbbbbb"))
    assert store.offer(DEVICE, "v1.0-5-gaaaaaaa") is None  # five commits past the tag
    assert store.offer(DEVICE, "v1.1") is None  # a later tag
    assert store.offer(DEVICE, "v1.0-1-gccccccc").version == "v1.0-3-gbbbbbbb"
    assert store.offer(DEVICE, "v0.9-12-gddddddd").version == "v1.0-3-gbbbbbbb"


@pytest.mark.parametrize(
    "running,staged,expected",
    [
        ("1.0", "1.1", None),
        ("1.1", "1.1", "current"),
        ("1.2", "1.1", "newer"),
        ("1.0-dirty", "1.1", "dev"),
        ("v1.0-5-gaaaaaaa", "v1.0-3-gbbbbbbb", "newer"),
        ("v1.0", "v1.0-3-gbbbbbbb", None),
        ("v2", "v1.9", "newer"),
        # bare commit hashes (a repo with no tags) cannot be ordered: offer it
        ("f39716b", "f2e27ca", None),
        ("1234567", "f2e27ca", None),
        # different tag schemes cannot be ordered either
        ("release-a-2-gaaaaaaa", "v1.0", None),
    ],
)
def test_why_an_image_would_not_be_taken(running, staged, expected):
    from jarvis.remote.firmware import refusal

    assert refusal(running, staged) == expected


def test_a_replaced_image_is_noticed(tmp_path):
    config = make_config(tmp_path)
    store = FirmwareStore(config.firmware_dir)
    put_image(config, fake_image("1.1"))
    assert store.get(DEVICE).version == "1.1"
    put_image(config, fake_image("1.2", payload=b"\x66" * 5000))
    assert store.get(DEVICE).version == "1.2"


def test_a_device_id_cannot_walk_out_of_the_firmware_folder(tmp_path):
    config = make_config(tmp_path)
    store = FirmwareStore(config.firmware_dir)
    (tmp_path / "secret.bin").write_bytes(fake_image("9.9"))
    assert store.get("../secret") is None


def test_a_device_id_ending_in_a_newline_is_not_a_device(tmp_path):
    """Known issue #12: the id comes from an HTTP header here, unchecked by hello."""
    config = make_config(tmp_path)
    put_image(config, fake_image("1.0"), device=DEVICE + "\n")  # so only the check can refuse it
    store = FirmwareStore(config.firmware_dir)
    assert store.get(DEVICE + "\n") is None


# -- hello carries the version --------------------------------------------


def test_hello_may_carry_the_running_version():
    ok, msg = P.validate_c2s(json.dumps(P.hello(TOKEN, DEVICE, fw="1.0")))
    assert ok and msg["fw"] == "1.0"
    ok, _ = P.validate_c2s(json.dumps({**P.hello(TOKEN, DEVICE), "fw": 3}))
    assert not ok
    ok, _ = P.validate_c2s(json.dumps({**P.hello(TOKEN, DEVICE), "fw": "x" * 33}))
    assert not ok


def test_a_newer_image_is_announced_after_ready(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        data = fake_image("1.1")
        put_image(config, data)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            import websockets

            async with websockets.connect(brain.url) as ws:
                edge = FakeEdge(ws)
                await edge.send(P.hello(TOKEN, DEVICE, fw="1.0"))
                await edge.expect("ready")
                event = await edge.expect("event")
                assert event["kind"] == "ota"
                assert event["data"] == {
                    "version": "1.1",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "path": P.FIRMWARE_PATH,
                }
        finally:
            await close(brain)

    run(scenario())


def test_nothing_is_announced_when_the_edge_runs_it_already(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            import websockets

            async with websockets.connect(brain.url) as ws:
                edge = FakeEdge(ws)
                await edge.send(P.hello(TOKEN, DEVICE, fw="1.1"))
                await edge.expect("ready")
                await edge.expect("state")
                with pytest.raises(asyncio.TimeoutError):
                    await edge.recv(timeout=0.3)
                assert "event" not in [m["type"] for m in edge.received]
        finally:
            await close(brain)

    run(scenario())


# -- GET /firmware --------------------------------------------------------


def test_the_image_is_served_to_its_own_device(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        data = fake_image("1.1")
        put_image(config, data)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            status, headers, body = await http_get(brain.port, headers=auth())
            assert status == 200
            assert body == data
            assert headers["content-length"] == str(len(data))
            assert headers["x-firmware-version"] == "1.1"
            assert headers["x-firmware-sha256"] == hashlib.sha256(data).hexdigest()
            assert headers["cache-control"] == "no-store"
        finally:
            await close(brain)

    run(scenario())


def test_a_bad_token_or_unknown_device_gets_nothing(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            status, _, body = await http_get(brain.port, headers=auth(token="wrong"))
            assert status == 401 and b"\xe9" not in body
            # the backoff that guards hello guards this door too
            status, _, _ = await http_get(brain.port, headers=auth())
            assert status == 429
        finally:
            await close(brain)

    run(scenario())


def test_no_credentials_at_all_is_refused(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            status, _, _ = await http_get(brain.port)
            assert status == 401
        finally:
            await close(brain)

    run(scenario())


def test_no_staged_image_is_not_found(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            status, _, _ = await http_get(brain.port, headers=auth())
            assert status == 404
        finally:
            await close(brain)

    run(scenario())


def test_the_websocket_still_answers_next_to_it(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            import websockets

            async with websockets.connect(brain.url) as ws:
                ready = await FakeEdge(ws).hello(speech=False)
                assert ready["type"] == "ready"
        finally:
            await close(brain)

    run(scenario())
