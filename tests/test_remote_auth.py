"""Auth and TLS on the edge link (M3 decision 9).

The link is assumed to be on the public internet. So: a per-device pre-shared
token compared in constant time, a `hello` that must arrive first and soon, and
a refusal to listen on a routable address in the clear.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import ssl
import subprocess
from dataclasses import replace

import pytest

from jarvis.remote import protocol as P
from jarvis.remote.server import RemoteLink, build_ssl_context
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeTranscriber, make_config


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


async def _close_code(brain, first=None, wait=0.0) -> int | None:
    """Connect, optionally send one message, and report how the brain closed."""
    import websockets

    try:
        async with websockets.connect(brain.url) as ws:
            if first is not None:
                await ws.send(json.dumps(first))
            if wait:
                await asyncio.sleep(wait)
            await asyncio.wait_for(ws.recv(), timeout=5)
    except Exception as exc:  # noqa: BLE001 — the close is the answer
        rcvd = getattr(exc, "rcvd", None)
        return getattr(rcvd, "code", None)
    return None


# -- the token -------------------------------------------------------------


def test_a_bad_token_is_refused_with_unauthorized(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            code = await _close_code(brain, P.hello("wrong", DEVICE))
            assert code == P.CLOSE.UNAUTHORIZED
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_an_unknown_device_gets_the_same_answer_as_a_bad_token(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            code = await _close_code(brain, P.hello(TOKEN, "someone-elses-pi"))
            assert code == P.CLOSE.UNAUTHORIZED
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_guessing_is_slowed_down(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            assert await _close_code(brain, P.hello("wrong", DEVICE)) == P.CLOSE.UNAUTHORIZED
            # the next attempt from the same address is refused before it is read
            assert await _close_code(brain, P.hello(TOKEN, DEVICE)) == P.CLOSE.UNAUTHORIZED
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_the_token_is_never_logged(tmp_path, caplog):
    import logging

    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            with caplog.at_level(logging.DEBUG):
                await _close_code(brain, P.hello("wrong", DEVICE))
                await _close_code(brain, P.hello(TOKEN, DEVICE))
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()
        assert not any(TOKEN in r.getMessage() for r in caplog.records)
        assert not any("wrong" in r.getMessage() for r in caplog.records)

    run(scenario())


# -- hello -----------------------------------------------------------------


def test_no_hello_within_the_timeout_closes_the_socket(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            code = await _close_code(brain, None, wait=0.0)
            assert code == P.CLOSE.BAD_PROTOCOL
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_something_other_than_hello_first_closes_the_socket(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            assert await _close_code(brain, P.speaking(True)) == P.CLOSE.BAD_PROTOCOL
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_junk_before_hello_closes_the_socket(tmp_path):
    async def scenario():
        import websockets

        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            code = None
            try:
                async with websockets.connect(brain.url) as ws:
                    await ws.send("{not json")
                    await asyncio.wait_for(ws.recv(), timeout=5)
            except Exception as exc:  # noqa: BLE001
                rcvd = getattr(exc, "rcvd", None)
                code = getattr(rcvd, "code", None)
            assert code == P.CLOSE.BAD_PROTOCOL
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_an_older_protocol_is_told_so_rather_than_failing_confusingly(tmp_path):
    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            old = {**P.hello(TOKEN, DEVICE), "protocol": P.PROTOCOL_VERSION - 1}
            assert await _close_code(brain, old) == P.CLOSE.BAD_PROTOCOL
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


# -- one edge per brain ----------------------------------------------------


def test_a_second_device_is_refused_while_one_is_connected(tmp_path):
    async def scenario():
        import websockets

        config = replace(
            make_config(tmp_path),
            edge_tokens={DEVICE: TOKEN, "kitchen": "other"},
        )
        brain = await Brain(config, transcriber=FakeTranscriber()).start(
            run_orchestrator=False
        )
        try:
            async with websockets.connect(brain.url) as ws:
                await ws.send(json.dumps(P.hello(TOKEN, DEVICE)))
                assert json.loads(await ws.recv())["type"] == "ready"
                code = await _close_code(brain, P.hello("other", "kitchen"))
                assert code == P.CLOSE.UNAUTHORIZED
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_the_same_device_reconnecting_replaces_its_old_socket(tmp_path):
    async def scenario():
        import websockets

        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            first = await websockets.connect(brain.url)
            await first.send(json.dumps(P.hello(TOKEN, DEVICE)))
            assert json.loads(await first.recv())["type"] == "ready"

            async with websockets.connect(brain.url) as second:
                await second.send(json.dumps(P.hello(TOKEN, DEVICE)))
                assert json.loads(await second.recv())["type"] == "ready"
                await asyncio.sleep(0.2)
                assert brain.link.connection is not None
            await first.close()
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


# -- behind a tunnel / reverse proxy --------------------------------------
#
# With Cloudflare Tunnel (or any reverse proxy) in front, every connection
# reaches the brain from 127.0.0.1. Keyed on that, the auth backoff cannot tell
# the Pi in the hall from someone hammering the public hostname — so a stranger
# guessing tokens locks out the real edge. The proxy's own header is the only
# thing that can tell them apart, and it is trusted only when the connection
# genuinely came from loopback.


async def _close_code_as(brain, ip, msg):
    import websockets

    try:
        async with websockets.connect(
            brain.url, additional_headers={"CF-Connecting-IP": ip}
        ) as ws:
            await ws.send(json.dumps(msg))
            await asyncio.wait_for(ws.recv(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        rcvd = getattr(exc, "rcvd", None)
        return getattr(rcvd, "code", None)
    return None


def _proxied_config(tmp_path):
    base = make_config(tmp_path)
    return replace(base, server=replace(base.server, trusted_proxy_header="CF-Connecting-IP"))


def test_a_flood_through_the_tunnel_cannot_lock_out_the_real_edge(tmp_path):
    async def scenario():
        brain = await Brain(_proxied_config(tmp_path)).start(run_orchestrator=False)
        try:
            # somebody guessing, repeatedly, from one address
            for _ in range(3):
                assert await _close_code_as(
                    brain, "203.0.113.5", P.hello("wrong", DEVICE)
                ) == P.CLOSE.UNAUTHORIZED
            # the real edge, from its own address, gets straight in
            import websockets

            async with websockets.connect(
                brain.url, additional_headers={"CF-Connecting-IP": "198.51.100.7"}
            ) as ws:
                await ws.send(json.dumps(P.hello(TOKEN, DEVICE)))
                assert json.loads(await ws.recv())["type"] == "ready"
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_the_guesser_is_still_slowed_down_per_address(tmp_path):
    async def scenario():
        brain = await Brain(_proxied_config(tmp_path)).start(run_orchestrator=False)
        try:
            assert await _close_code_as(brain, "203.0.113.5", P.hello("wrong", DEVICE)) == (
                P.CLOSE.UNAUTHORIZED
            )
            # same address again, now with the right token: still refused, because
            # that address is in its backoff
            assert await _close_code_as(brain, "203.0.113.5", P.hello(TOKEN, DEVICE)) == (
                P.CLOSE.UNAUTHORIZED
            )
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_the_proxy_header_is_ignored_unless_it_is_configured(tmp_path):
    """Default off. An unconfigured brain that believed the header would let
    anyone reset their own backoff by inventing an address."""

    async def scenario():
        brain = await Brain(make_config(tmp_path)).start(run_orchestrator=False)
        try:
            assert await _close_code_as(brain, "203.0.113.5", P.hello("wrong", DEVICE)) == (
                P.CLOSE.UNAUTHORIZED
            )
            # a different claimed address, but the same real peer: still blocked
            assert await _close_code_as(brain, "198.51.100.7", P.hello(TOKEN, DEVICE)) == (
                P.CLOSE.UNAUTHORIZED
            )
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


def test_the_header_is_only_believed_from_a_loopback_peer(tmp_path):
    """The rule that makes trusting a header safe: the tunnel runs on this
    machine. A header from anywhere else is somebody else's claim about
    themselves."""
    from jarvis.remote.server import RemoteServer

    config = _proxied_config(tmp_path)
    server = RemoteServer(config, RemoteLink(config))

    class FakeWS:
        def __init__(self, peer, header=None):
            self.remote_address = (peer, 1234)
            self.request = type("R", (), {"headers": {"CF-Connecting-IP": header} if header else {}})()

    assert server.client_key(FakeWS("127.0.0.1", "203.0.113.5")) == "203.0.113.5"
    assert server.client_key(FakeWS("::1", "203.0.113.5")) == "203.0.113.5"
    # not loopback: the header is somebody's claim, not the proxy's
    assert server.client_key(FakeWS("192.168.1.50", "203.0.113.5")) == "192.168.1.50"
    assert server.client_key(FakeWS("127.0.0.1")) == "127.0.0.1"
    # a chain takes the first entry, which is the original client
    assert server.client_key(FakeWS("127.0.0.1", "203.0.113.5, 10.0.0.1")) == "203.0.113.5"


def test_a_proxy_in_a_container_can_be_trusted_by_address(tmp_path):
    """`cloudflared` in Docker reaches the brain over the bridge, not loopback,
    so loopback-only trust would silently ignore its header — and silently is
    the bad part: the backoff would quietly lump every client together again."""
    from jarvis.remote.server import RemoteServer

    base = make_config(tmp_path)
    config = replace(
        base,
        server=replace(
            base.server,
            trusted_proxy_header="CF-Connecting-IP",
            trusted_proxy_peers=("172.16.0.0/12", "192.168.1.5"),
        ),
    )
    server = RemoteServer(config, RemoteLink(config))

    class FakeWS:
        def __init__(self, peer, header=None):
            self.remote_address = (peer, 1234)
            self.request = type("R", (), {"headers": {"CF-Connecting-IP": header} if header else {}})()

    # the docker bridge, and a named proxy host on the LAN
    assert server.client_key(FakeWS("172.17.0.1", "203.0.113.5")) == "203.0.113.5"
    assert server.client_key(FakeWS("192.168.1.5", "203.0.113.5")) == "203.0.113.5"
    # loopback is always trusted, listed or not
    assert server.client_key(FakeWS("127.0.0.1", "203.0.113.5")) == "203.0.113.5"
    # anything else is not
    assert server.client_key(FakeWS("192.168.1.99", "203.0.113.5")) == "192.168.1.99"
    assert server.client_key(FakeWS("203.0.113.9", "10.0.0.1")) == "203.0.113.9"


def test_loopback_stays_trusted_with_no_peers_configured(tmp_path):
    from jarvis.remote.server import RemoteServer

    config = _proxied_config(tmp_path)
    assert config.server.trusted_proxy_peers == ()
    server = RemoteServer(config, RemoteLink(config))

    class FakeWS:
        remote_address = ("127.0.0.1", 1)
        request = type("R", (), {"headers": {"CF-Connecting-IP": "203.0.113.5"}})()

    assert server.client_key(FakeWS()) == "203.0.113.5"


def test_a_mistyped_trusted_peer_fails_loudly_at_startup(tmp_path):
    """A typo here means the backoff quietly stops distinguishing clients, which
    is exactly the failure nobody would notice. Refuse to start instead."""
    from jarvis.remote.server import RemoteServer

    base = make_config(tmp_path)
    config = replace(
        base,
        server=replace(
            base.server,
            trusted_proxy_header="CF-Connecting-IP",
            trusted_proxy_peers=("172.16.0.0/12", "not-an-address"),
        ),
    )
    with pytest.raises(ValueError) as exc:
        RemoteServer(config, RemoteLink(config))
    assert "not-an-address" in str(exc.value)
    assert "trusted_proxy_peers" in str(exc.value)


def test_trusted_peers_without_a_header_name_do_nothing(tmp_path):
    from jarvis.remote.server import RemoteServer

    base = make_config(tmp_path)
    config = replace(
        base, server=replace(base.server, trusted_proxy_peers=("172.16.0.0/12",))
    )
    server = RemoteServer(config, RemoteLink(config))

    class FakeWS:
        remote_address = ("172.17.0.1", 1)
        request = type("R", (), {"headers": {"CF-Connecting-IP": "203.0.113.5"}})()

    assert server.client_key(FakeWS()) == "172.17.0.1"


def test_a_successful_hello_clears_that_address_backoff(tmp_path):
    async def scenario():
        brain = await Brain(_proxied_config(tmp_path)).start(run_orchestrator=False)
        try:
            import websockets

            async with websockets.connect(
                brain.url, additional_headers={"CF-Connecting-IP": "198.51.100.7"}
            ) as ws:
                await ws.send(json.dumps(P.hello(TOKEN, DEVICE)))
                assert json.loads(await ws.recv())["type"] == "ready"
            await asyncio.sleep(0.1)
            assert "198.51.100.7" not in brain.server._backoff
        finally:
            brain.ws_server.close()
            await brain.ws_server.wait_closed()

    run(scenario())


# -- TLS -------------------------------------------------------------------


def test_a_routable_address_without_tls_is_refused(tmp_path):
    config = replace(
        make_config(tmp_path),
        server=replace(make_config(tmp_path).server, host="0.0.0.0", port=8765),
    )
    with pytest.raises(RuntimeError) as exc:
        build_ssl_context(config)
    message = str(exc.value)
    assert "TLS" in message
    assert "allow_insecure" in message and "127.0.0.1" in message


def test_loopback_without_tls_is_fine_because_a_proxy_terminates_it(tmp_path):
    config = make_config(tmp_path)  # host is 127.0.0.1
    assert build_ssl_context(config) is None


def test_allow_insecure_is_permitted_and_says_so_loudly(tmp_path, caplog):
    import logging

    base = make_config(tmp_path)
    config = replace(base, server=replace(base.server, host="0.0.0.0", allow_insecure=True))
    with caplog.at_level(logging.WARNING):
        assert build_ssl_context(config) is None
    assert any("allow_insecure" in r.getMessage() for r in caplog.records)
    assert any("clear" in r.getMessage() for r in caplog.records)


def test_a_certificate_and_key_give_a_tls_context(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    cert, key = tmp_path / "jarvis.pem", tmp_path / "jarvis.key"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/CN=jarvis.test", "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True,
    )
    base = make_config(tmp_path)
    config = replace(
        base,
        server=replace(base.server, host="0.0.0.0", tls_cert=str(cert), tls_key=str(key)),
    )
    assert config.server.tls_enabled is True
    assert isinstance(build_ssl_context(config), ssl.SSLContext)


def test_a_missing_certificate_file_fails_at_startup_not_mid_conversation(tmp_path):
    base = make_config(tmp_path)
    config = replace(
        base,
        server=replace(base.server, tls_cert=str(tmp_path / "nope.pem"), tls_key=str(tmp_path / "nope.key")),
    )
    with pytest.raises((FileNotFoundError, ssl.SSLError, OSError)):
        build_ssl_context(config)
