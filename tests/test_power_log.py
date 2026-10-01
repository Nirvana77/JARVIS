"""The watch's power log, fetched to the brain and read there.

The watch keeps a CSV per day on its SD card (``jarvis-edge/main/powerlog.c``):
a row every 10 s and one per event. The brain asks for it with the edge tool
``send_power_log``, saying how much of each file it already has; the watch
sends only the rest, as ``file`` messages, then ``done``. The brain keeps them
in ``data/remote/power/<device_id>/`` and reads them: time per mode, how much
of it asleep, the battery's drain per mode, restarts and drops.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from jarvis.core.context import Context
from jarvis.remote import powerlog as PL
from jarvis.remote import protocol as P
from jarvis.remote.server import PowerFetch
from jarvis.skills.builtin import watch_power
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeEdge, make_config

HEADER = PL.HEADER


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


def row(t, up, mode, mv, sleep="", charging=0, usb=0, pct=80, event="", wifi="up", link=1):
    sleeps = "5" if sleep != "" else ""
    ev = f'"{event}"' if event else ""
    return (
        f"2026-10-01 {t},{up},{mode},off,{wifi},-60,max,{link},{pct},{mv},{charging},{usb},"
        f"{sleep},{sleeps},120,{ev}\n"
    )


def day_csv():
    rows = [row("08:00:00", 1, "awake", 4000, event="boot (power on), firmware abc")]
    up, mv = 1, 4000
    # 10 minutes awake, 1 mV a sample (60 mV/10 min = 360 mV/h), barely asleep
    for _ in range(60):
        up += 10
        mv -= 1
        rows.append(row("08:00:00", up, "awake", mv, sleep=5))
    rows.append(row("08:10:00", up, "watch", mv, event="watch mode (idle timeout)"))
    # 20 minutes in watch mode, 1 mV a minute (60 mV/h), mostly asleep
    for i in range(120):
        up += 10
        if i % 6 == 5:
            mv -= 1
        rows.append(row("08:20:00", up, "watch", mv, sleep=95))
    rows.append(row("08:30:00", up, "watch", mv, event="Wi-Fi: dropped from atwork (reason 8)"))
    # a restart, then 5 minutes charging (not counted as drain)
    rows.append(row("08:31:00", 1, "awake", mv, event="boot (brownout), firmware abc"))
    up = 1
    for _ in range(30):
        up += 10
        mv += 2
        rows.append(row("08:31:00", up, "awake", mv, sleep=0, charging=1, usb=1, pct=81))
    return HEADER + "".join(rows)


# -- protocol ----------------------------------------------------------------


def test_file_chunks_and_done():
    ok, msg = P.validate_c2s(P.file_chunk("power", "2026-10-01.csv", 0, "a,b\n", eof=True))
    assert ok, msg
    ok, msg = P.validate_c2s(P.file_done("power", files=2))
    assert ok, msg


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "file", "kind": "power", "name": "../../etc/passwd", "offset": 0, "b64": ""},
        {"type": "file", "kind": "power", "name": "2026-10-01.txt", "offset": 0, "b64": ""},
        {"type": "file", "kind": "secrets", "name": "2026-10-01.csv", "offset": 0, "b64": ""},
        {"type": "file", "kind": "power", "name": "2026-10-01.csv", "offset": -1, "b64": ""},
        {"type": "file", "kind": "power", "name": "2026-10-01.csv", "offset": 0,
         "b64": "x" * (P.MAX_FILE_CHUNK + 4)},
    ],
)
def test_bad_file_chunks_are_refused(bad):
    ok, _ = P.validate_c2s(bad)
    assert not ok


# -- reading it --------------------------------------------------------------


def test_the_summary(tmp_path):
    path = tmp_path / "2026-10-01.csv"
    path.write_text(day_csv(), encoding="utf-8")
    s = PL.summarize(PL.read_rows([path]))
    assert s.modes["awake"].seconds == pytest.approx(600 + 300, abs=20)
    assert s.modes["watch"].seconds == pytest.approx(1200, abs=20)
    assert s.modes["watch"].sleep_pct == pytest.approx(95, abs=1)
    # drain only while on battery: awake ~360 mV/h, watch ~60 mV/h
    assert s.modes["awake"].mv_per_h == pytest.approx(360, rel=0.1)
    assert s.modes["watch"].mv_per_h == pytest.approx(60, rel=0.2)
    assert [b.reason for b in s.boots] == ["power on", "brownout"]
    assert s.wifi_drops == 1
    assert s.charging_now is True
    text = PL.report(s)
    assert "brownout" in text and "watch" in text
    line = PL.spoken(s)
    assert "watch mode" in line and "%" in line


def test_rows_without_a_clock_and_junk_are_survived(tmp_path):
    path = tmp_path / "nodate.csv"
    path.write_text(HEADER + "-,5,awake,on,up,,none,1,80,4000,0,0,,,100,\"boot (crash), firmware x\"\n"
                    "garbage line\n", encoding="utf-8")
    s = PL.summarize(PL.read_rows([path]))
    assert [b.reason for b in s.boots] == ["crash"]


# -- fetching it -------------------------------------------------------------

POWER_TOOL = {"name": "send_power_log", "description": "Send the power log", "examples": [], "params": {}}


async def connected(brain, tools=(POWER_TOOL,)):
    import websockets

    ws = await websockets.connect(brain.url)
    edge = FakeEdge(ws)
    msg = P.hello(TOKEN, DEVICE)
    msg["tools"] = list(tools)
    await edge.send(msg)
    await edge.expect("ready")
    await edge.expect("state")  # sent on connect: not the answer to anything later
    return ws, edge


async def watch_sends(edge, files: dict[str, str], chunk=1000):
    """The watch's side: answer send_power_log with what the brain lacks."""
    call = await edge.expect("call", timeout=10)
    assert call["tool"] == "send_power_log"
    have = call["args"]["have"]
    await edge.send(P.result(call["id"], True, ""))
    sent = 0
    for name, text in files.items():
        start = have.get(name, 0)
        if start > len(text):
            start = 0
        if start == len(text):
            continue
        for at in range(start, len(text), chunk):
            part = text[at : at + chunk]
            await edge.send(P.file_chunk("power", name, at, part, eof=at + chunk >= len(text)))
        sent += 1
    await edge.send(P.file_done("power", files=sent))
    return have


def test_a_fetch_brings_only_what_is_new(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            text = day_csv()
            first = asyncio.ensure_future(watch_sends(edge, {"2026-10-01.csv": text[:5000]}))
            result = await brain.server.fetch_power_log()
            assert (await first) == {}
            assert result.status == "ok" and result.files == 1
            folder = config.remote_dir / "power" / DEVICE
            assert (folder / "2026-10-01.csv").read_text() == text[:5000]

            second = asyncio.ensure_future(watch_sends(edge, {"2026-10-01.csv": text}))
            result = await brain.server.fetch_power_log()
            assert (await second) == {"2026-10-01.csv": 5000}  # it said what it had
            assert (folder / "2026-10-01.csv").read_text() == text
            await ws.close()
        finally:
            await brain.stop()

    run(scenario())


def test_a_resent_chunk_overwrites_from_its_offset(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            folder = config.remote_dir / "power" / DEVICE
            folder.mkdir(parents=True)
            (folder / "2026-10-01.csv").write_text("aaaaXXXX")
            await edge.send(P.file_chunk("power", "2026-10-01.csv", 4, "bbbb", eof=True))
            await edge.send(P.file_chunk("power", "2026-10-01.csv", 99, "gap", eof=True))
            await edge.send(P.control(P.CONTROL.MIC, on=True))
            await edge.expect("state")
            assert (folder / "2026-10-01.csv").read_text() == "aaaabbbb"
            await ws.close()
        finally:
            await brain.stop()

    run(scenario())


def test_no_edge_an_old_edge_and_a_stalled_one(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            assert (await brain.server.fetch_power_log()).status == "offline"
            ws, _ = await connected(brain, tools=())
            assert (await brain.server.fetch_power_log()).status == "unsupported"
            await ws.close()
            await asyncio.sleep(0.1)
            ws, edge = await connected(brain)

            async def answers_but_never_sends():
                call = await edge.expect("call")
                await edge.send(P.result(call["id"], True, ""))

            task = asyncio.ensure_future(answers_but_never_sends())
            result = await brain.server.fetch_power_log(timeout_s=0.5)
            assert result.status == "timeout"
            await task
            await ws.close()
        finally:
            await brain.stop()

    run(scenario())


def test_power_over_http(tmp_path):
    import urllib.request

    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            sender = asyncio.ensure_future(watch_sends(edge, {"2026-10-01.csv": day_csv()}))

            def get():
                req = urllib.request.Request(
                    f"http://127.0.0.1:{brain.port}/power?day=2026-10-01",
                    headers={"Authorization": f"Bearer {TOKEN}", "X-Jarvis-Device": DEVICE},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, resp.read().decode()

            status, body = await asyncio.to_thread(get)
            await sender
            assert status == 200
            assert "fetched 1 file" in body and "brownout" in body
            await ws.close()
        finally:
            await brain.stop()

    run(scenario())


def test_the_cli_url(tmp_path):
    from jarvis.app import power_request

    config = make_config(tmp_path)
    config = dataclasses.replace(config, server=dataclasses.replace(config.server, port=8765))
    url, headers = power_request(config, day="2026-10-01", fetch=False)
    assert url == "http://127.0.0.1:8765/power?day=2026-10-01&fetch=0"
    assert headers["X-Jarvis-Device"] == DEVICE


# -- the spoken skill ----------------------------------------------------------


class FakeEdges:
    def __init__(self, result):
        self.result = result

    def power_report(self):
        return self.result


def ctx(config, tmp_path, edges):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None, edges=edges)


@pytest.mark.parametrize(
    "result,words",
    [
        (PowerFetch("ok", files=1, spoken="Today the watch slept 95% of watch mode."), "slept"),
        (PowerFetch("offline", spoken="Last I heard, it slept 90%."), "isn't connected"),
        (PowerFetch("unsupported"), "can't send"),
        (PowerFetch("offline"), "isn't connected"),
    ],
)
def test_the_skill(config, tmp_path, result, words):
    line = watch_power.run(ctx(config, tmp_path, FakeEdges(result)))
    assert words in line.lower()


def test_the_skill_without_a_remote_link(config, tmp_path):
    assert "isn't connected" in watch_power.run(ctx(config, tmp_path, None)).lower()
