"""Pairing: a device without a token gets one, once a person says so.

A device with no token sends ``pair`` (its X25519 public key) instead of
``hello``. The brain answers with its own; both derive the same key and the
same 6-digit code, and the device shows the code. Nothing happens until
somebody with access to the brain confirms that code (``python -m jarvis pair
482913``): that is the approval, and it also proves nobody sat in between.
Then the brain sends a fresh token, sealed with the shared key, and the device
keeps it and reconnects with an ordinary ``hello``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import os
import stat

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from jarvis.remote import pairing
from jarvis.remote import protocol as P
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeEdge, make_config


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


def keypair():
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private, public


# -- the derivation -------------------------------------------------------------


def test_both_sides_derive_the_same_key_and_code():
    watch_priv, watch_pub = keypair()
    brain_priv, brain_pub = keypair()
    a = pairing.derive(watch_priv.exchange(pairing.public_key(brain_pub)), "watch", watch_pub, brain_pub)
    b = pairing.derive(brain_priv.exchange(pairing.public_key(watch_pub)), "watch", watch_pub, brain_pub)
    assert a == b
    assert len(a.key) == 32 and len(a.code) == 6 and a.code.isdigit()


def test_someone_in_between_gets_a_different_code():
    watch_priv, watch_pub = keypair()
    brain_priv, brain_pub = keypair()
    mallory_priv, mallory_pub = keypair()
    # the watch thinks it talks to the brain, but got mallory's key
    watch_side = pairing.derive(watch_priv.exchange(pairing.public_key(mallory_pub)), "watch", watch_pub, mallory_pub)
    brain_side = pairing.derive(brain_priv.exchange(pairing.public_key(mallory_pub)), "watch", mallory_pub, brain_pub)
    assert watch_side.code != brain_side.code


def test_the_sealed_token_opens_only_with_the_key():
    secret = pairing.derive(os.urandom(32), "watch", b"a" * 32, b"b" * 32)
    nonce, box = pairing.seal(secret.key, "watch", "the-token")
    assert pairing.open_box(secret.key, "watch", nonce, box) == "the-token"
    with pytest.raises(Exception):
        pairing.open_box(secret.key, "other", nonce, box)  # bound to the device id


def test_a_known_vector():
    """Pinned, so the watch's C (mbedtls) can be checked against the same
    numbers: shared = 00 01 .. 1f, device "watch", keys 32 x 01 and 32 x 02."""
    s = pairing.derive(bytes(range(32)), "watch", bytes([1]) * 32, bytes([2]) * 32)
    assert (s.key.hex(), s.code) == pairing.KNOWN_VECTOR
    assert s.code == "079006"


# -- protocol --------------------------------------------------------------------


def test_pair_is_a_first_message():
    ok, msg = P.validate_c2s(P.pair(DEVICE, b"k" * 32))
    assert ok, msg
    for bad in (
        {"type": "pair", "protocol": 1, "device_id": DEVICE, "key": "short"},
        {"type": "pair", "protocol": 1, "device_id": "../x", "key": base64.b64encode(b"k" * 32).decode()},
        {"type": "pair", "device_id": DEVICE, "key": base64.b64encode(b"k" * 32).decode()},
    ):
        assert not P.validate_c2s(bad)[0]


# -- the token store -------------------------------------------------------------


def test_the_store_keeps_tokens_private(tmp_path):
    store = pairing.TokenStore(tmp_path)
    store.set("watch", "t1")
    assert pairing.TokenStore(tmp_path).get("watch") == "t1"
    mode = stat.S_IMODE((tmp_path / "tokens.json").stat().st_mode)
    assert mode & 0o077 == 0  # nobody else on the machine reads them
    assert store.devices() == ["watch"]
    assert store.remove("watch") and store.get("watch") is None


def test_the_admin_secret_is_made_once(tmp_path):
    first = pairing.admin_secret(tmp_path)
    assert pairing.admin_secret(tmp_path) == first and len(first) >= 32
    assert stat.S_IMODE((tmp_path / "admin.token").stat().st_mode) & 0o077 == 0


# -- the server ------------------------------------------------------------------


async def start_pairing(brain, device_id="newwatch"):
    """The device's side, up to the code it would show."""
    import websockets

    ws = await websockets.connect(brain.url)
    edge = FakeEdge(ws)
    priv, pub = keypair()
    await edge.send(P.pair(device_id, pub))
    msg = await edge.expect("pairing")
    brain_pub = base64.b64decode(msg["key"])
    secret = pairing.derive(priv.exchange(pairing.public_key(brain_pub)), device_id, pub, brain_pub)
    return ws, edge, secret


def test_pairing_end_to_end(tmp_path):
    import websockets

    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge, secret = await start_pairing(brain)
            assert brain.server.approve_pairing("000000" if secret.code != "000000" else "111111") is None
            assert brain.server.approve_pairing(secret.code) == "newwatch"
            paired = await edge.expect("paired")
            token = pairing.open_box(
                secret.key, "newwatch", base64.b64decode(paired["nonce"]), base64.b64decode(paired["box"])
            )
            await asyncio.sleep(0.1)
            # and now it is a device like any other
            ws2 = await websockets.connect(brain.url)
            edge2 = FakeEdge(ws2)
            await edge2.send(P.hello(token, "newwatch"))
            assert (await edge2.expect("ready"))["type"] == "ready"
            await ws2.close()
            assert pairing.TokenStore(config.remote_dir).get("newwatch") == token
        finally:
            await brain.stop()

    run(scenario())


def test_an_unapproved_request_expires(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        brain.server.PAIR_TIMEOUT_S = 0.3
        try:
            ws, edge, secret = await start_pairing(brain)
            await asyncio.wait_for(ws.wait_closed(), 3)
            assert ws.close_code == P.CLOSE.PAIR_EXPIRED
            assert brain.server.approve_pairing(secret.code) is None  # too late
            assert pairing.TokenStore(config.remote_dir).get("newwatch") is None
        finally:
            await brain.stop()

    run(scenario())


def test_a_removed_device_is_refused(tmp_path):
    import websockets

    async def scenario():
        config = make_config(tmp_path)
        pairing.TokenStore(config.remote_dir).set("newwatch", "stored-token")
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws = await websockets.connect(brain.url)
            edge = FakeEdge(ws)
            await edge.send(P.hello("stored-token", "newwatch"))
            await edge.expect("ready")
            await ws.close()
            pairing.TokenStore(config.remote_dir).remove("newwatch")
            await asyncio.sleep(0.1)
            ws = await websockets.connect(brain.url)
            await FakeEdge(ws).send(P.hello("stored-token", "newwatch"))
            await asyncio.wait_for(ws.wait_closed(), 3)
            assert ws.close_code == P.CLOSE.UNAUTHORIZED
        finally:
            await brain.stop()

    run(scenario())


def test_approval_over_http_needs_the_admin_secret(tmp_path):
    import urllib.error
    import urllib.request

    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge, secret = await start_pairing(brain)

            def get(auth):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{brain.port}/pair?code={secret.code}",
                    headers={"Authorization": f"Bearer {auth}"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        return resp.status, resp.read().decode()
                except urllib.error.HTTPError as exc:
                    return exc.code, ""

            # a device's token is not enough: approving is the brain's owner's call
            assert (await asyncio.to_thread(get, TOKEN))[0] == 401
            await asyncio.sleep(2.2)  # the refusal's backoff
            status, body = await asyncio.to_thread(get, pairing.admin_secret(config.remote_dir))
            assert status == 200 and "newwatch" in body
            await edge.expect("paired")
        finally:
            await brain.stop()

    run(scenario())


def test_the_cli_requests(tmp_path):
    from jarvis.app import pair_request

    config = make_config(tmp_path)
    config = dataclasses.replace(config, server=dataclasses.replace(config.server, port=8765))
    url, headers = pair_request(config, "482 913")
    assert url == "http://127.0.0.1:8765/pair?code=482913"
    assert headers == {"Authorization": f"Bearer {pairing.admin_secret(config.remote_dir)}"}


def test_the_device_list(tmp_path, capsys):
    from jarvis.app import devices

    config = make_config(tmp_path)
    pairing.TokenStore(config.remote_dir).set("newwatch", "t")
    assert devices(config) == 0
    out = capsys.readouterr().out
    assert "newwatch" in out and "paired" in out and DEVICE in out  # .env ones too
    assert devices(config, remove="newwatch") == 0
    assert pairing.TokenStore(config.remote_dir).get("newwatch") is None
