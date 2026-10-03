"""Edge tools: an edge declares what it can do, and JARVIS calls it.

The watch lists its tools in ``hello`` (name, description, example phrases,
typed params). The brain keeps the list per device, registers each tool as an
``edge`` skill (so the classifier learns its examples), and a skill run sends
``call`` to the edge and speaks the ``result``. ``notify`` is the same call
started by something other than a turn: ``python -m jarvis notify "..."`` or
``GET /notify``, queued while the edge is away.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from jarvis.core.context import Context
from jarvis.nlu import slots
from jarvis.remote import protocol as P
from jarvis.remote.server import EdgeControl, ToolResult
from jarvis.skills.edge import EdgeTools, edge_skills
from jarvis.skills.registry import Registry
from tests.remote_harness import DEVICE, TOKEN, Brain, FakeEdge, make_config


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


TIMER = {
    "name": "set_timer",
    "description": "Start a timer on the watch",
    "examples": ["set a timer for 5 minutes", "remind me in 20 minutes to take the pizza out"],
    "params": {
        "seconds": {"type": "duration", "required": True},
        "label": {"type": "text"},
    },
}
FIND = {
    "name": "find_watch",
    "description": "Make the watch ring so it can be found",
    "examples": ["find my watch", "where is my watch"],
    "params": {},
}
NOTIFY = {
    "name": "notify",
    "description": "Show a notification on the watch",
    "examples": [],
    "params": {"text": {"type": "text", "required": True}},
}
TOOLS = [TIMER, FIND, NOTIFY]


def hello(tools=TOOLS, **kw):
    msg = P.hello(TOKEN, DEVICE, **kw)
    if tools is not None:
        msg["tools"] = tools
    return msg


# -- protocol ----------------------------------------------------------------


def test_hello_may_carry_tools():
    ok, msg = P.validate_c2s(hello())
    assert ok, msg
    assert [t["name"] for t in msg["tools"]] == ["set_timer", "find_watch", "notify"]


def test_hello_without_tools_is_still_fine():
    ok, msg = P.validate_c2s(hello(tools=None))
    assert ok and "tools" not in msg


@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        [{"name": "Set Timer", "description": "x", "examples": [], "params": {}}],
        [{"name": "set_timer\n", "description": "x", "examples": [], "params": {}}],
        [{"name": "t", "description": "x" * 300, "examples": [], "params": {}}],
        [{"name": "t", "description": "x", "examples": ["y" * 200], "params": {}}],
        [{"name": "t", "description": "x", "examples": [], "params": {"p": {"type": "file"}}}],
        [{"name": "t", "description": "x", "examples": [], "params": {"p": "duration"}}],
        [dict(FIND, name=f"t{i}") for i in range(P.MAX_TOOLS + 1)],
        [FIND, FIND],
    ],
)
def test_bad_tool_lists_are_refused(bad):
    ok, err = P.validate_c2s(hello(tools=bad))
    assert not ok and "tool" in err


def test_result_and_call():
    ok, msg = P.validate_c2s({"type": "result", "id": "c1", "ok": True, "say": "Timer set."})
    assert ok, msg
    ok, err = P.validate_c2s({"type": "result", "id": "c1", "ok": "yes"})
    assert not ok
    ok, err = P.validate_c2s({"type": "result", "ok": True})
    assert not ok
    assert P.call("c1", "set_timer", {"seconds": 60}) == {
        "type": "call", "id": "c1", "tool": "set_timer", "args": {"seconds": 60},
    }


# -- typed slots -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("set a timer for 5 minutes", 300),
        ("set a timer for an hour", 3600),
        ("timer for 1 hour and 30 minutes", 5400),
        ("start a timer for 90 seconds", 90),
        ("set a timer for half an hour", 1800),
        ("set a timer for two minutes", 120),
        ("set a 3 minute timer", 180),
        ("remind me in 20 minutes to take the pizza out", 1200),
        ("a timer for one and a half hours", 5400),
    ],
)
def test_durations(text, seconds):
    params = slots.extract_typed(TIMER["params"], text)
    assert params["seconds"] == seconds


@pytest.mark.parametrize(
    "text,label",
    [
        ("remind me in 20 minutes to take the pizza out", "take the pizza out"),
        ("remind me to call mom in an hour", "call mom"),
        ("set a timer for 5 minutes", None),
        # "for" names the thing, once the duration (and its own "for") is out
        ("cancel the timer for the oven", "the oven"),
        ("set a timer for the pizza for 10 minutes", "the pizza"),
        ("cancel the 5 minute timer", None),
    ],
)
def test_text_after_the_duration(text, label):
    params = slots.extract_typed(TIMER["params"], text)
    assert params.get("label") == label


def test_no_duration_no_param():
    assert "seconds" not in slots.extract_typed(TIMER["params"], "set a timer")


def test_numbers():
    spec = {"level": {"type": "number"}}
    assert slots.extract_typed(spec, "set the brightness to 40") == {"level": 40}
    assert slots.extract_typed(spec, "brightness to seventy") == {"level": 70}


# -- the store and the skills ------------------------------------------------


def test_the_store_says_when_the_list_changed(tmp_path):
    store = EdgeTools(tmp_path)
    assert store.save(DEVICE, TOOLS) is True
    assert store.save(DEVICE, TOOLS) is False  # a reconnect with the same firmware
    assert store.save(DEVICE, [FIND]) is True
    assert [t["name"] for t in EdgeTools(tmp_path).tools(DEVICE)] == ["find_watch"]


def test_edge_skills_are_manifests_with_origin_edge(tmp_path):
    store = EdgeTools(tmp_path)
    store.save(DEVICE, TOOLS)
    skills = {s.MANIFEST.name: s for s in edge_skills(store)}
    assert set(skills) == {"set_timer", "find_watch", "notify"}
    timer = skills["set_timer"].MANIFEST
    assert timer.origin == "edge"
    assert timer.examples == TIMER["examples"]
    assert timer.required_params == ["seconds"]


def test_the_registry_picks_them_up_and_builtins_win(config, tmp_path):
    config = dataclasses.replace(config, data_dir=tmp_path)
    clash = dict(FIND, name="search")  # a builtin's name
    EdgeTools(config.remote_dir).save(DEVICE, [TIMER, clash])
    registry = Registry.discover(config)
    assert "set_timer" in registry
    assert registry.manifest("set_timer").origin == "edge"
    assert registry.manifest("search").origin == "builtin"


class FakeEdges:
    def __init__(self, result=None):
        self.result = result or ToolResult("ok", say="Timer set for 5 minutes.")
        self.calls = []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return self.result


def ctx(config, tmp_path, edges):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None, edges=edges)


def skill(name, tmp_path):
    store = EdgeTools(tmp_path / "remote")
    store.save(DEVICE, TOOLS)
    return {s.MANIFEST.name: s for s in edge_skills(store)}[name]


def test_an_edge_skill_calls_the_edge_and_says_its_answer(config, tmp_path):
    edges = FakeEdges()
    line = skill("set_timer", tmp_path).run(ctx(config, tmp_path, edges), seconds=300)
    assert edges.calls == [("set_timer", {"seconds": 300})]
    assert line == "Timer set for 5 minutes."


def test_an_edge_skill_asks_for_what_it_needs(config, tmp_path):
    edges = FakeEdges()
    line = skill("set_timer", tmp_path).run(ctx(config, tmp_path, edges))
    assert edges.calls == []
    assert "how long" in line.lower()


@pytest.mark.parametrize(
    "result,words",
    [
        (ToolResult("offline"), "isn't connected"),
        (ToolResult("timeout"), "didn't answer"),
        (ToolResult("failed", say="No room for another timer."), "no room"),
        (ToolResult("ok", say=""), "done"),
    ],
)
def test_an_edge_skill_explains_failures(config, tmp_path, result, words):
    line = skill("find_watch", tmp_path).run(ctx(config, tmp_path, FakeEdges(result)))
    assert words in line.lower()


def test_an_edge_skill_without_a_remote_link(config, tmp_path):
    line = skill("find_watch", tmp_path).run(ctx(config, tmp_path, None))
    assert "isn't connected" in line.lower()


# -- the server --------------------------------------------------------------


async def connected(brain, tools=TOOLS):
    import websockets

    ws = await websockets.connect(brain.url)
    edge = FakeEdge(ws)
    await edge.send(hello(tools=tools))
    await edge.expect("ready")
    return ws, edge


async def close(brain):
    await brain.stop()


async def answer_calls(edge, ok=True, say="Done it."):
    """The edge's side: answer every `call` that arrives."""
    while True:
        msg = await edge.expect("call", timeout=30)
        await edge.send({"type": "result", "id": msg["id"], "ok": ok, "say": say})


def test_hello_tools_are_kept_and_reported_once(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        changes = []
        brain.server.on_tools_changed = lambda device: changes.append(device)
        try:
            ws, _ = await connected(brain)
            await ws.close()
            await asyncio.sleep(0.1)
            ws, _ = await connected(brain)  # same list: nothing to relearn
            await ws.close()
            await asyncio.sleep(0.1)
            assert changes == [DEVICE]
            assert [t["name"] for t in brain.server.tools.tools(DEVICE)] == [
                "set_timer", "find_watch", "notify",
            ]
        finally:
            await close(brain)

    run(scenario())


def test_a_call_goes_out_and_its_result_comes_back(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            responder = asyncio.ensure_future(answer_calls(edge, say="Timer set."))
            control = EdgeControl(brain.server)
            result = await asyncio.to_thread(control.call, "set_timer", {"seconds": 60})
            assert result == ToolResult("ok", say="Timer set.")
            call = next(m for m in edge.received if m["type"] == "call")
            assert call["tool"] == "set_timer" and call["args"] == {"seconds": 60}
            responder.cancel()
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_an_unknown_tool_and_no_edge(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            assert (await brain.server.call_tool("find_watch", {})).status == "offline"
            ws, _ = await connected(brain, tools=[FIND])
            assert (await brain.server.call_tool("set_timer", {})).status == "unsupported"
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_a_silent_edge_times_out_and_a_dropped_one_fails_fast(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, _ = await connected(brain)
            result = await brain.server.call_tool("find_watch", {}, timeout_s=0.3)
            assert result.status == "timeout"
            pending = asyncio.ensure_future(brain.server.call_tool("find_watch", {}, timeout_s=10))
            await asyncio.sleep(0.1)
            await ws.close()
            assert (await asyncio.wait_for(pending, 3)).status == "offline"
        finally:
            await close(brain)

    run(scenario())


def test_a_late_or_stray_result_is_ignored(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            await edge.send({"type": "result", "id": "nobody-asked", "ok": True})
            await edge.send(P.control(P.CONTROL.MIC, on=True))
            await edge.expect("state")  # still talking to us
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


# -- notify ------------------------------------------------------------------


def test_notify_reaches_a_connected_edge(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            ws, edge = await connected(brain)
            responder = asyncio.ensure_future(answer_calls(edge, say=""))
            assert await brain.server.notify("The build is done") == "sent"
            call = next(m for m in edge.received if m["type"] == "call")
            assert call["tool"] == "notify" and call["args"] == {"text": "The build is done"}
            responder.cancel()
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_notify_waits_for_an_edge_that_is_away(tmp_path):
    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)
        try:
            assert await brain.server.notify("first") == "queued"
            assert await brain.server.notify("second") == "queued"
            ws, edge = await connected(brain)
            texts = []
            for _ in range(2):
                call = await edge.expect("call")
                texts.append(call["args"]["text"])
                await edge.send({"type": "result", "id": call["id"], "ok": True})
            assert texts == ["first", "second"]
            await ws.close()
        finally:
            await close(brain)

    run(scenario())


def test_notify_over_http_needs_the_device_token(tmp_path):
    import urllib.error
    import urllib.request

    async def scenario():
        config = make_config(tmp_path)
        brain = await Brain(config).start(run_orchestrator=False)

        def get(token):
            req = urllib.request.Request(
                f"http://127.0.0.1:{brain.port}/notify?text=Build%20done",
                headers={"Authorization": f"Bearer {token}", "X-Jarvis-Device": DEVICE},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status, resp.read().decode()
            except urllib.error.HTTPError as exc:
                return exc.code, ""

        try:
            status, _ = await asyncio.to_thread(get, "wrong")
            assert status == 401
            await asyncio.sleep(2.2)  # the refusal's backoff
            status, body = await asyncio.to_thread(get, TOKEN)
            assert status == 200 and "queued" in body
            assert list(brain.server.outbox[DEVICE]) == ["Build done"]
        finally:
            await close(brain)

    run(scenario())


def test_the_cli_url(tmp_path):
    from jarvis.app import notify_request

    config = make_config(tmp_path)
    config = dataclasses.replace(config, server=dataclasses.replace(config.server, port=8765))
    url, headers = notify_request(config, "Build done & tested")
    assert url == "http://127.0.0.1:8765/notify?text=Build+done+%26+tested"
    assert headers == {"Authorization": f"Bearer {TOKEN}", "X-Jarvis-Device": DEVICE}


# -- the orchestrator --------------------------------------------------------


class ManifestRegistry:
    """A registry that knows one edge skill, and records dispatches."""

    def __init__(self, tmp_path):
        self.calls = []
        store = EdgeTools(tmp_path / "remote")
        store.save(DEVICE, TOOLS)
        self._skills = {s.MANIFEST.name: s for s in edge_skills(store)}

    def manifest(self, name):
        return self._skills[name].MANIFEST

    def manifests(self):
        return [s.MANIFEST for s in self._skills.values()]

    def names(self):
        return sorted(self._skills)

    def __contains__(self, name):
        return name in self._skills

    def dispatch(self, label, params):
        self.calls.append((label, params))
        return "ok"


def test_an_edge_label_gets_typed_params(tmp_path):
    from jarvis.core.orchestrator import Orchestrator

    registry = ManifestRegistry(tmp_path)
    orch = Orchestrator.__new__(Orchestrator)
    orch.registry = registry
    orch._slot_extract = slots.extract
    params = orch._params_for("set_timer", "remind me in 20 minutes to take the pizza out")
    assert params == {"seconds": 1200, "label": "take the pizza out"}
    assert orch._params_for("search", "search for cats") == {"query": "cats"}


def test_new_tools_are_learned_in_the_background(tmp_path):
    from tests.remote_harness import FakeNLU

    async def scenario():
        config = make_config(tmp_path)
        brain = Brain(config)
        trained = []

        async def fake_train(examples):
            trained.append({e.label for e in examples})
            return "ok", (None, FakeNLU({"find my watch": ("find_watch", 0.9)}))

        orch = brain.orchestrator
        orch._train_and_load = fake_train
        orch.registry = Registry.discover(config)
        EdgeTools(config.remote_dir).save(DEVICE, TOOLS)
        await orch.refresh_skills()
        assert trained and {"set_timer", "find_watch"} <= trained[0]
        assert "notify" not in trained[0]  # no examples: never routed by voice
        nlu, registry = orch._staged
        assert "set_timer" in registry
        await orch._merge_gate()
        assert orch.registry is registry

    run(scenario())


def test_a_missing_param_is_asked_for_and_the_answer_used(tmp_path):
    from jarvis.core.orchestrator import Orchestrator

    async def scenario():
        orch = Orchestrator.__new__(Orchestrator)
        orch.registry = ManifestRegistry(tmp_path)
        asked, spoken = [], []

        async def ask(prompt):
            asked.append(prompt)
            return answers.pop(0)

        async def speak(line):
            spoken.append(line)

        orch._ask, orch._speak = ask, speak
        orch.persona = type("P", (), {"phrase": staticmethod(lambda t: t)})()

        answers = ["five minutes"]
        params = await orch._fill_missing("set_timer", {})
        assert params == {"seconds": 300}
        assert asked == ["How long, sir?"]

        answers = ["banana"]
        assert await orch._fill_missing("set_timer", {}) is None
        assert spoken and "catch" in spoken[-1]

        # nothing missing, or not an edge skill: no question
        asked.clear()
        assert await orch._fill_missing("set_timer", {"seconds": 60}) == {"seconds": 60}
        assert await orch._fill_missing("search", {"query": "x"}) == {"query": "x"}
        assert asked == []

    run(scenario())


# -- names ("the pizza timer") -------------------------------------------------

NAMED = {
    "seconds": {"type": "duration"},
    "name": {"type": "name"},
    "label": {"type": "text"},
}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("set a pizza timer for 10 minutes", {"seconds": 600, "name": "pizza"}),
        ("set a timer called pasta for 8 minutes", {"seconds": 480, "name": "pasta"}),
        ("start a tea timer for 3 minutes", {"seconds": 180, "name": "tea"}),
        ("set a 10 minute pizza timer", {"seconds": 600, "name": "pizza"}),
        ("set a timer named egg for 6 minutes", {"seconds": 360, "name": "egg"}),
        ("cancel the pizza timer", {"name": "pizza"}),
        ("stop the timer called pasta", {"name": "pasta"}),
        ("cancel the pasta sauce timer", {"name": "pasta sauce"}),
        # no name: these are not names
        ("set a timer for 5 minutes", {"seconds": 300}),
        ("set a 3 minute timer", {"seconds": 180}),
        ("cancel the timer", {}),
        ("cancel my timer", {}),
        ("cancel all timers", {}),
        ("remind me in 20 minutes to take the pizza out", {"seconds": 1200, "label": "take the pizza out"}),
    ],
)
def test_names(text, expected):
    assert slots.extract_typed(NAMED, text) == expected


def test_name_is_a_param_type_an_edge_may_declare():
    tool = dict(TIMER, params={"name": {"type": "name"}})
    ok, msg = P.validate_c2s(hello(tools=[tool]))
    assert ok, msg
