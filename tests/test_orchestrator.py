"""Orchestrator state machine, exercised with fake audio/NLU components."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import threading

from jarvis.config import load_config
from jarvis.core.orchestrator import Orchestrator
from jarvis.nlu.corpus import intent_meta


# -- fakes ------------------------------------------------------------------

class FakeMic:
    """``script`` is a list of bools for successive ``record_utterance`` calls:
    True -> a spoken utterance, False -> silence. Past the end reads as silence.

    ``talking_reads`` makes the first N ``read()`` calls return loud frames so the
    orchestrator's onset detector fires (simulates the user talking over TTS)."""

    def __init__(self, script=(True,), talking_reads=0):
        self._script = list(script)
        self._i = 0
        self._talking_reads = talking_reads
        self._reads = 0
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def read(self, timeout=None):
        self._reads += 1
        if self._reads <= self._talking_reads:
            return np.full(1280, 6000, dtype=np.int16)
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
    def __init__(self, on_dispatch=None, result="done", manifests=()):
        self.calls = []
        self.on_dispatch = on_dispatch
        self.result = result
        self.raises = None
        self._manifests = list(manifests)

    def dispatch(self, label, params):
        self.calls.append((label, params))
        if self.on_dispatch:
            self.on_dispatch()
        if self.raises:
            raise self.raises
        return f"{self.result}:{label}:{params}"

    # -- M2: the factory flows only need read access -------------------------
    def names(self):
        return [m.name for m in self._manifests]

    def manifests(self):
        return list(self._manifests)

    def manifest(self, name):
        return next(m for m in self._manifests if m.name == name)

    def __contains__(self, name):
        return name in self.names()


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


def test_unknown_does_not_auto_offer_to_teach(orch):
    """A garbled/unrecognized command should just say so — it must not
    proactively launch into "shall I learn how to do that, sir?"."""
    orch.standby = False
    orch.claude_client = type("FakeAvailableClient", (), {"available": True})()
    asyncio.run(orch.handle("unknown", "flibber"))
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]


def test_low_confidence_meta_action_falls_back_to_unknown(orch):
    """Regression: "flip of corn" (a garbled "flip a coin") once classified as
    revert_skill at 0.43 confidence — barely above the general "unknown" cutoff
    — and launched the whole revert dialog instead of just saying it didn't
    catch the command."""
    orch.standby = False
    asyncio.run(orch.handle("revert_skill", "flip of corn", confidence=0.43))
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]


def test_high_confidence_meta_action_still_proceeds(orch):
    orch.standby = False
    asyncio.run(orch.handle("revert_skill", "revert the timer skill", confidence=0.9))
    # it reached the real revert_skill flow (which then reports no matching
    # skill on the empty FakeRegistry) rather than bailing out as "unknown"
    assert "<unknown>" not in orch._persona.spoken
    assert any("skills" in s.lower() for s in orch._persona.spoken)


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


def test_ask_does_not_print_heard_by_default(orch, capsys):
    """A teach/edit_skill/revert_skill clarifying question's answer isn't
    classified (no intent to show) — unlike a normal command it stays quiet
    unless verbose logging is on."""
    import logging

    from jarvis.core import orchestrator as orchestrator_module

    assert not orchestrator_module.log.isEnabledFor(logging.INFO)  # default (WARNING)
    asyncio.run(orch._ask("What should I call this skill?"))
    assert 'heard' not in capsys.readouterr().out


def test_ask_prints_heard_when_verbose(orch, capsys):
    import logging

    from jarvis.core import orchestrator as orchestrator_module

    original = orchestrator_module.log.level
    orchestrator_module.log.setLevel(logging.INFO)
    try:
        asyncio.run(orch._ask("What should I call this skill?"))
    finally:
        orchestrator_module.log.setLevel(original)
    assert 'heard   : "search black holes"' in capsys.readouterr().out


def test_voice_barge_in_abandons_the_in_flight_command(orch, capsys):
    # opt-in extra (off by default); force it on for this test
    orch._interrupter.enabled_barge = True
    # transcription is slow (0.5s block); the user starts talking over it, so the
    # onset detector fires and the decode is abandoned
    orch.stt = FakeSTT(text="search black holes", block=threading.Event())
    orch.mic = FakeMic(script=[True, False], talking_reads=12)
    asyncio.run(orch.run())
    out = capsys.readouterr().out
    assert "barge-in" in out
    # only the post-barge-in command dispatched (once); the abandoned one didn't
    assert orch.registry.calls == [("search", {"query": "black holes"})]
    assert all(f.done() for f in orch._interrupter._draining)


def test_enter_cancels_transcription_and_stays_listening(orch):
    orch._interrupter.enabled_barge = False  # isolate the Enter path
    orch.stt = FakeSTT(block=threading.Event())  # 0.5s block
    orch.mic = FakeMic(script=[True, False])

    async def press_enter_soon():
        await asyncio.sleep(0.05)
        orch._interrupter._interrupt.set()

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
    # two utterances, no talking-over -> two sequential commands, then standby
    orch.mic = FakeMic(script=[True, True])
    asyncio.run(orch.run())
    assert orch.registry.calls == [
        ("search", {"query": "black holes"}),
        ("search", {"query": "black holes"}),
    ]
    assert "<standby>" in orch._persona.spoken


# -- M2: merge gate + skill factory degradation ------------------------------

def test_merge_gate_does_not_swap_while_busy(orch):
    """PRD M2 verification: a staged replacement must not be promoted mid-turn."""
    new_nlu, new_registry = object(), object()
    orch.state = "acting"
    orch._staged = (new_nlu, new_registry)
    orch._pending_announcement = "I've learned 'timer', sir."

    asyncio.run(orch._merge_gate())

    assert orch.nlu is not new_nlu
    assert orch.registry is not new_registry
    assert orch._staged is not None
    assert orch._pending_announcement is not None
    assert "timer" not in " ".join(orch._persona.spoken)


def test_merge_gate_swaps_and_announces_once_idle(orch):
    new_nlu, new_registry = object(), object()
    orch.state = "idle"
    orch._staged = (new_nlu, new_registry)
    orch._pending_announcement = "I've learned 'timer', sir."

    asyncio.run(orch._merge_gate())

    assert orch.nlu is new_nlu
    assert orch.registry is new_registry
    assert orch._staged is None
    assert orch._pending_announcement is None
    assert "I've learned 'timer', sir." in orch._persona.spoken


def test_merge_gate_is_a_noop_with_nothing_staged(orch):
    old_nlu, old_registry = orch.nlu, orch.registry
    orch.state = "idle"
    asyncio.run(orch._merge_gate())
    assert orch.nlu is old_nlu
    assert orch.registry is old_registry


def test_teach_degrades_gracefully_without_a_factory(orch):
    orch.standby = False
    orch.claude_client = None
    asyncio.run(orch.handle("teach", "learn how to set a timer"))
    assert "factory" in orch._persona.spoken[-1].lower()
    assert orch.registry.calls == []


def test_revert_skill_does_not_require_a_factory(orch):
    orch.standby = False
    orch.claude_client = None
    asyncio.run(orch.handle("revert_skill", "revert the timer skill"))
    # no factory needed — it just can't find a matching learned skill on the
    # bare FakeRegistry, so it should ask, get no reply, and give up quietly
    assert orch.registry.calls == []
