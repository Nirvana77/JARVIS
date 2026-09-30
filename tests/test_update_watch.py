"""'Update the watch': a spoken command that sends the staged firmware.

The brain already offers a staged image when an edge connects
(``test_remote_firmware.py``). This is the same offer on demand, while the edge
is connected — announced only once the reply has finished playing, so the
download never talks over JARVIS saying it is starting.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from jarvis.core.context import Context
from jarvis.remote import protocol as P
from jarvis.remote.server import EdgeControl, FirmwareUpdate
from jarvis.skills.builtin import update_watch
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeEdge, make_config
from tests.test_remote_firmware import close, fake_image, put_image


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


# -- the skill ---------------------------------------------------------------


class FakeEdges:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def update_firmware(self):
        self.calls += 1
        return self.result


def ctx(config, tmp_path, edges=None):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None, edges=edges)


def test_the_skill_says_it_is_updating(config, tmp_path):
    edges = FakeEdges(FirmwareUpdate("sent", version="abc123", device_id="watch"))
    line = update_watch.run(ctx(config, tmp_path, edges))
    assert edges.calls == 1
    assert "abc123" in line and "updat" in line.lower()


@pytest.mark.parametrize(
    "status,words",
    [
        ("current", "already"),
        ("none", "no new firmware"),
        ("unsupported", "can't update"),
        ("offline", "isn't connected"),
    ],
)
def test_the_skill_explains_why_nothing_happened(config, tmp_path, status, words):
    line = update_watch.run(ctx(config, tmp_path, FakeEdges(FirmwareUpdate(status, version="abc123"))))
    assert words in line.lower()


def test_without_a_remote_link_the_skill_says_so(config, tmp_path):
    # all-in-one and text mode: there is no edge to update
    line = update_watch.run(ctx(config, tmp_path, None))
    assert "isn't connected" in line.lower()


# -- the server --------------------------------------------------------------


async def connected(brain, fw="1.0", speech=True):
    import websockets

    ws = await websockets.connect(brain.url)
    edge = FakeEdge(ws)
    await edge.send(P.hello(TOKEN, DEVICE, fw=fw))
    await edge.expect("ready")
    if speech:
        await edge.send(P.control(P.CONTROL.SET_SPEECH, on=True, sample_rate=16000))
        await edge.expect("ready")
    return ws, edge


async def no_event(edge, wait=0.3):
    with pytest.raises(asyncio.TimeoutError):
        while True:
            msg = await edge.recv(timeout=wait)
            assert msg["type"] != "event", msg


def test_the_update_is_announced_after_the_reply_has_played(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain, fw="1.1")  # nothing on connect
            put_image(config, fake_image("1.2", payload=b"\x77" * 3000))
            result = await brain.server.update_firmware()
            assert result == FirmwareUpdate("sent", version="1.2", device_id=DEVICE)
            await no_event(edge)  # the reply is still to be spoken
            await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id="s1"))
            event = await edge.expect("event")
            assert event["kind"] == "ota" and event["data"]["version"] == "1.2"
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_an_edge_without_speech_gets_it_at_once(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain, fw="1.0", speech=False)
            put_image(config, fake_image("1.1"))
            result = await brain.server.update_firmware()
            assert result.status == "sent"
            event = await edge.expect("event")
            assert event["data"]["version"] == "1.1"
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_nothing_is_sent_when_it_runs_that_version(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        put_image(config, fake_image("1.1"))
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain, fw="1.1")
            assert await brain.server.update_firmware() == FirmwareUpdate(
                "current", version="1.1", device_id=DEVICE
            )
            await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id="s1"))
            await no_event(edge)
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_no_image_an_old_edge_and_no_edge(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            assert (await brain.server.update_firmware()).status == "offline"
            ws, _ = await connected(brain, fw="1.0")
            assert (await brain.server.update_firmware()).status == "none"
            await ws.close()
            await asyncio.sleep(0.1)
            ws, _ = await connected(brain, fw=None)  # never said what it runs
            put_image(config, fake_image("1.1"))
            assert (await brain.server.update_firmware()).status == "unsupported"
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_a_skill_thread_can_ask_through_edge_control(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain, fw="1.0", speech=False)
            put_image(config, fake_image("1.1"))
            control = EdgeControl(brain.server)
            # skills run in a worker thread (orchestrator: asyncio.to_thread)
            result = await asyncio.to_thread(control.update_firmware)
            assert result.status == "sent"
            await edge.expect("event")
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_edge_control_before_the_loop_runs_is_offline(tmp_path):
    from jarvis.remote.server import RemoteLink, RemoteServer

    config = make_config(tmp_path)
    server = RemoteServer(config, RemoteLink(config), transcriber=object())
    assert EdgeControl(server).update_firmware().status == "offline"


# -- the NLU knows the new skill without a manual retrain --------------------


def test_a_skill_the_model_has_never_seen_triggers_a_retrain(config, tmp_path, monkeypatch):
    from jarvis import app
    from jarvis.skills.registry import BUILTIN_PACKAGE, Registry

    cfg = dataclasses.replace(config, data_dir=tmp_path)
    vdir = cfg.nlu_model_dir / "v1"
    vdir.mkdir(parents=True)
    (vdir / "head.joblib").write_bytes(b"")  # what makes a version count as trained
    registry = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    labels = [n for n in registry.names() if n != "update_watch"]
    (vdir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    retrained = []
    monkeypatch.setattr(
        app, "rebuild_nlu",
        lambda c, r: retrained.append(1) or type("R", (), {"version": 2})(),
    )
    assert app.ensure_nlu(cfg, registry) == 2
    assert retrained == [1]

    (vdir / "labels.json").write_text(json.dumps(registry.names()), encoding="utf-8")
    retrained.clear()
    assert app.ensure_nlu(cfg, registry) == 1
    assert retrained == []


def test_a_reconnect_keeps_the_new_session(tmp_path):
    """The watch reconnects before the brain has noticed the old socket died:
    the old handler's cleanup runs *after* the new session is in place, and
    must not take the new one with it — or "update the watch" says it isn't
    connected while it is (seen on the real brain, 2026-09-30)."""

    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            old_ws, _ = await connected(brain, fw="1.0", speech=False)
            new_ws, edge = await connected(brain, fw="1.0", speech=False)
            # the brain closes the old socket; give its handler time to finish
            for _ in range(50):
                if old_ws.close_code is not None:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            assert DEVICE in brain.server.sessions
            put_image(config, fake_image("1.1"))
            assert (await brain.server.update_firmware()).status == "sent"
            await edge.expect("event")
            await new_ws.close()
        finally:
            await close(brain)

    run(scenario())
