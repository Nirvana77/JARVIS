"""Orchestrator state machine, exercised with fake audio/NLU components."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jarvis.config import load_config
from jarvis.core.orchestrator import Orchestrator
from jarvis.nlu.corpus import intent_meta


# -- fakes ------------------------------------------------------------------

class FakeMic:
    def __init__(self, utterance=b"x"):
        self._utterance = utterance
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def read(self, timeout=None):
        return np.zeros(1280, dtype=np.int16)

    def record_utterance(self, *a, **k):
        if self._utterance is None:
            return np.zeros(0, dtype=np.float32)
        return np.ones(16000, dtype=np.float32) * 0.1


class FakeWake:
    def __init__(self, fire_after=1):
        self._calls = 0
        self._fire_after = fire_after

    def reset(self):
        pass

    def triggered(self, frame):
        self._calls += 1
        return self._calls >= self._fire_after


class FakeSTT:
    def __init__(self, text="search black holes"):
        self.text = text

    def transcribe(self, audio):
        return self.text


class FakeNLU:
    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return self.mapping.get(text, ("unknown", 0.1))


class FakePersona:
    def __init__(self):
        self.spoken = []

    def line(self, event, default=""):
        return f"<{event}>"

    def phrase(self, text):
        return text


class FakeTTS:
    def __init__(self, persona):
        self.persona = persona

    def say(self, text):
        self.persona.spoken.append(text)


class FakeRegistry:
    def __init__(self, on_dispatch=None, result="done"):
        self.calls = []
        self.on_dispatch = on_dispatch
        self.result = result
        self.raises = None

    def dispatch(self, label, params):
        self.calls.append((label, params))
        if self.on_dispatch:
            self.on_dispatch()
        if self.raises:
            raise self.raises
        return f"{self.result}:{label}:{params}"


@pytest.fixture
def orch():
    persona = FakePersona()
    o = Orchestrator(
        config=load_config(),
        wake=FakeWake(),
        mic=FakeMic(),
        stt=FakeSTT(),
        tts=FakeTTS(persona),
        nlu=FakeNLU(
            {
                "search black holes": ("search", 0.9),
                "thanks": ("thanks", 0.9),
                "go to sleep": ("goodbye", 0.9),
                "shut down": ("shutdown", 0.9),
                "hey there": ("greeting", 0.9),
                "flibber": ("unknown", 0.1),
            }
        ),
        persona=persona,
        registry=FakeRegistry(),
        intent_meta=intent_meta(),
    )
    o._persona = persona
    return o


# -- handle() ---------------------------------------------------------------

def test_skill_dispatch_when_awake(orch):
    orch.standby = False
    asyncio.run(orch.handle("search", "search black holes"))
    assert orch.registry.calls == [("search", {"query": "black holes"})]
    assert orch._persona.spoken  # a line was produced


def test_actions_ignored_while_in_standby_until_woken(orch):
    orch.standby = True
    asyncio.run(orch.handle("search", "search black holes"))
    assert orch.registry.calls == []  # ignored

    asyncio.run(orch.handle("greeting", "hey there"))
    assert orch.standby is False
    assert "<greeting>" in orch._persona.spoken

    asyncio.run(orch.handle("search", "search black holes"))
    assert orch.registry.calls == [("search", {"query": "black holes"})]


def test_goodbye_returns_to_standby(orch):
    orch.standby = False
    asyncio.run(orch.handle("goodbye", "go to sleep"))
    assert orch.standby is True
    assert "<standby>" in orch._persona.spoken


def test_shutdown_stops_the_loop(orch):
    orch.standby = False
    orch.running = True
    asyncio.run(orch.handle("shutdown", "shut down"))
    assert orch.running is False
    assert "<shutdown>" in orch._persona.spoken


def test_unknown_speaks_the_unknown_line(orch):
    orch.standby = False
    asyncio.run(orch.handle("unknown", "flibber"))
    assert orch.registry.calls == []
    assert "<unknown>" in orch._persona.spoken


def test_none_action_speaks_canned_response(orch):
    orch.standby = False
    asyncio.run(orch.handle("thanks", "thanks"))
    assert orch.registry.calls == []
    assert orch._persona.spoken and orch._persona.spoken[-1]  # a canned reply


def test_skill_error_speaks_error_line(orch):
    orch.standby = False
    orch.registry.raises = RuntimeError("boom")
    asyncio.run(orch.handle("search", "search black holes"))
    assert "<error>" in orch._persona.spoken


# -- full run() loop --------------------------------------------------------

def test_run_loop_wake_to_dispatch(orch):
    orch.registry = FakeRegistry(on_dispatch=orch.stop)
    asyncio.run(orch.run())
    assert orch.registry.calls == [("search", {"query": "black holes"})]
    assert orch.mic.started is False  # stopped cleanly
    assert orch.state == "idle"
