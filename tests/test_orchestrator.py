"""Orchestrator state machine, exercised with fake audio/NLU components."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import threading

from jarvis.config import load_config
from jarvis.core.orchestrator import Orchestrator, _Cancelled
from jarvis.nlu.corpus import intent_meta


# -- fakes ------------------------------------------------------------------

class FakeMic:
    """``script`` is a list of bools: True -> a spoken utterance, False -> silence.
    Values past the end read as silence."""

    def __init__(self, script=(True,)):
        self._script = list(script)
        self._i = 0
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def read(self, timeout=None):
        return np.zeros(1280, dtype=np.int16)

    def drain(self):
        pass

    def record_utterance(self, *a, **k):
        has_speech = self._script[self._i] if self._i < len(self._script) else False
        self._i += 1
        if not has_speech:
            return np.zeros(0, dtype=np.float32)
        return np.ones(16000, dtype=np.float32) * 0.1


class FakeWake:
    """Fires ``fires`` times, then calls ``stop`` (if set) so run() can exit."""

    def __init__(self, fires=1):
        self.calls = 0
        self.fires = fires
        self.stop = None

    def reset(self):
        pass

    def triggered(self, frame):
        self.calls += 1
        if self.calls <= self.fires:
            return True
        if self.stop:
            self.stop()
        return False


class FakeSTT:
    def __init__(self, text="search black holes", block=None):
        self.text = text
        self.block = block  # a threading.Event; transcribe waits on it
        self.last_avg_logprob = -0.3

    def transcribe(self, audio, cancel_event=None):
        if self.block is not None:
            self.block.wait(0.5)
        return self.text


class FakeNLU:
    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, text):
        return self.mapping.get(text, ("unknown", 0.1))

    def explain(self, text):
        from jarvis.nlu.classifier import Prediction

        label, conf = self.mapping.get(text, ("unknown", 0.1))
        return Prediction(label, conf, [(label, conf)], 1.0)


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
    o.wake.stop = o.stop
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


def test_stays_awake_for_followups_then_announces_standby(orch):
    # one command, then a silent follow-up window
    orch.mic = FakeMic(script=[True, False])
    asyncio.run(orch.run())
    assert orch.registry.calls == [("search", {"query": "black holes"})]
    assert "<standby>" in orch._persona.spoken  # deferred standby announcement
    assert orch.standby is True


def test_transcript_and_intent_are_printed(orch, capsys):
    orch.standby = False
    orch.mic = FakeMic(script=[True, False])
    asyncio.run(orch.run())
    out = capsys.readouterr().out
    assert 'heard   : "search black holes"' in out
    assert "intent  : search" in out


def test_voice_barge_in_replaces_the_in_flight_command(orch, capsys):
    # transcription is slow; a fresh utterance arrives while it runs
    orch.stt = FakeSTT(text="search black holes", block=threading.Event())  # 0.5s block
    orch.mic = FakeMic(script=[True, True, False])  # cmd, barge-in, then silence
    asyncio.run(orch.run())
    # the first (abandoned) transcription never dispatched; only the barge-in did
    assert orch.registry.calls == [("search", {"query": "black holes"})]
    assert "barge-in" in capsys.readouterr().out
    assert orch._draining == [] or all(f.done() for f in orch._draining)


def test_enter_cancels_transcription_and_stays_listening(orch):
    orch.barge_in = False  # isolate the Enter path
    orch.stt = FakeSTT(block=threading.Event())  # 0.5s block
    orch.mic = FakeMic(script=[True, False])

    async def press_enter_soon():
        await asyncio.sleep(0.05)
        orch._interrupt.set()

    async def scenario():
        await asyncio.gather(orch.run(), press_enter_soon())

    asyncio.run(scenario())
    assert orch.registry.calls == []            # nothing dispatched
    assert "<standby>" in orch._persona.spoken  # session ended cleanly


def test_record_utterance_honours_stop_event():
    from jarvis.audio.capture import Microphone

    ev = threading.Event()
    ev.set()
    m = Microphone(16000, 1280)
    m.read = lambda timeout=None: np.zeros(1280, dtype=np.int16)
    assert len(m.record_utterance(max_seconds=2, stop_event=ev)) == 0


def test_two_commands_in_one_wake_session(orch):
    # silence during each transcription's barge-in window -> two real commands
    orch.mic = FakeMic(script=[True, False, True, False])
    asyncio.run(orch.run())
    assert orch.registry.calls == [
        ("search", {"query": "black holes"}),
        ("search", {"query": "black holes"}),
    ]
    assert "<standby>" in orch._persona.spoken
