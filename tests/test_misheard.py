"""M4 — misheard-command reasoning (`PRD/jarvis-2026-rebuild.md`, Milestone 4).

Where the orchestrator used to say "didn't catch that" unconditionally, it now
asks the local reasoner whether the transcript is a plausible mishearing of
something JARVIS knows, confirms the guess by voice, and only then runs the
*corrected* text through the real NLU and dispatches it.

The reasoner is a fake with a canned reply, so nothing here needs Ollama.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from jarvis.config import load_config
from jarvis.core import mishear
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.reasoner import Reasoner
from jarvis.nlu.corpus import intent_meta
from jarvis.skills.contract import SkillManifest
from tests.test_orchestrator import (
    FakeMic,
    FakeNLU,
    FakePersona,
    FakeRegistry,
    FakeTTS,
    FakeWake,
)


# -- fakes ------------------------------------------------------------------

class FakeReasoner:
    """A deterministic canned guess. ``reply`` may be an exception to raise."""

    def __init__(self, reply="flip a coin", available=True):
        self.reply = reply
        self.available = available
        self.calls = []

    def generate(self, system, prompt, **options):
        self.calls.append((system, prompt, options))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class ScriptedSTT:
    """Successive ``transcribe`` calls return successive ``replies`` — the
    command first, then the answer to "did you mean". Past the end: silence."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.last_avg_logprob = -0.3

    def transcribe(self, audio, cancel_event=None):
        return self.replies.pop(0) if self.replies else ""


class RecordingNLU(FakeNLU):
    """``FakeNLU`` that remembers what it was asked. Like the real classifier
    it does not care about case."""

    def __init__(self, mapping):
        super().__init__(mapping)
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        return super().predict(text.lower())

    def explain(self, text):
        self.seen.append(text)
        return super().explain(text.lower())


class DidYouMeanPersona(FakePersona):
    """``FakePersona``, but the confirm keeps its ``{guess}`` slot so a test
    can see which phrase was offered."""

    def line(self, event, default=""):
        if event == "did_you_mean":
            return "<did_you_mean {guess}>"
        return super().line(event, default)


COIN = SkillManifest(
    name="flip_a_coin",
    description="Flip one or more coins and report heads or tails.",
    examples=["Flip a coin.", "heads or tails?", "toss a coin for me", "flip three coins"],
    origin="learned",
)

NLU_MAPPING = {
    "flip a coin": ("flip_a_coin", 0.9),
    "search black holes": ("search", 0.9),
    "go to sleep": ("goodbye", 0.9),
    "revert the timer skill": ("revert_skill", 0.9),
    "revert the coin skill": ("revert_skill", 0.45),  # recognised, but not sure enough
}


def make_orch(reasoner, answers=(), config=None):
    """An awake orchestrator whose next spoken replies are ``answers``."""
    persona = DidYouMeanPersona()
    o = Orchestrator(
        config=config or load_config(),
        wake=FakeWake(),
        mic=FakeMic(script=[True] * len(answers)),
        stt=ScriptedSTT(answers),
        tts=FakeTTS(persona),
        nlu=RecordingNLU(NLU_MAPPING),
        persona=persona,
        registry=FakeRegistry(manifests=[COIN]),
        intent_meta=intent_meta(),
        reasoner=reasoner,
    )
    o._persona = persona
    o.wake.stop = o.stop
    o.standby = False
    return o


def confirms(orch):
    return [s for s in orch._persona.spoken if s.startswith("<did_you_mean")]


# -- reasoner absent: behaviour unchanged ---------------------------------------

def test_without_a_reasoner_unknown_is_unchanged():
    orch = make_orch(reasoner=None)
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_an_unavailable_reasoner_is_never_asked():
    reasoner = FakeReasoner(available=False)
    orch = make_orch(reasoner)
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert reasoner.calls == []
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_low_confidence_meta_action_without_a_reasoner_is_unchanged():
    orch = make_orch(reasoner=None)
    asyncio.run(orch.handle("revert_skill", "flip of corn", confidence=0.43))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_it_can_be_switched_off_with_ollama_still_running():
    """`[reasoner] correct_misheard = false`: the persona keeps its rewrite,
    the "did you mean" step is skipped."""
    config = load_config()
    config = dataclasses.replace(
        config, reasoner=dataclasses.replace(config.reasoner, correct_misheard=False)
    )
    reasoner = FakeReasoner("flip a coin")
    orch = make_orch(reasoner, answers=["yes"], config=config)
    asyncio.run(orch.handle("unknown", "flip a corn"))
    # (M4.5 may still ask it to plan or answer — not to guess a mishearing)
    assert [system for system, _p, _o in reasoner.calls if system == mishear.SYSTEM] == []
    assert orch._persona.spoken == ["<unknown>"]


# -- a plausible correction, confirmed ------------------------------------------

def test_a_confirmed_correction_is_classified_and_dispatched():
    orch = make_orch(FakeReasoner("flip a coin"), answers=["yes"])
    asyncio.run(orch.handle("unknown", "flip a corn"))

    assert confirms(orch) == ["<did_you_mean flip a coin>"]
    assert "flip a coin" in orch.nlu.seen           # the real NLU decided what it is
    assert [label for label, _ in orch.registry.calls] == ["flip_a_coin"]
    assert "<unknown>" not in orch._persona.spoken


def test_the_corrected_text_is_dispatched_not_the_raw_reply():
    """The reasoner's reply is cleaned up — quotes, a label, a full stop, a
    trailing explanation — and it is that phrase, re-classified by the real
    NLU, that is offered and run. Slots come from it too."""
    raw = 'Intended command: "Search black holes."\n\n(The speaker said "surge".)'
    orch = make_orch(FakeReasoner(raw), answers=["yes please"])
    asyncio.run(orch.handle("unknown", "surge black holes"))

    assert raw not in orch.nlu.seen
    assert orch.nlu.seen[-1].lower() == "search black holes"
    assert orch.registry.calls == [("search", {"query": "black holes"})]


def test_nothing_is_dispatched_before_the_confirmation():
    """Never act on a guess directly: with no answer at all, nothing runs."""
    orch = make_orch(FakeReasoner("flip a coin"), answers=[])
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert confirms(orch) == ["<did_you_mean flip a coin>"]
    assert orch.registry.calls == []
    assert orch._persona.spoken[-1] == "<unknown>"


@pytest.mark.parametrize("reply", ["I'm unsure", "that's incorrect", "yesterday", "not sure, no"])
def test_only_a_clear_yes_runs_the_guess(reply):
    """"unsure" has "sure" in it and "incorrect" has "correct": matching the
    yes-words anywhere in the reply would run a guess nobody agreed to."""
    orch = make_orch(FakeReasoner("flip a coin"), answers=[reply])
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert confirms(orch) == ["<did_you_mean flip a coin>"]
    assert orch.registry.calls == []
    assert orch._persona.spoken[-1] == "<unknown>"


def test_low_confidence_meta_action_gets_the_same_second_look():
    """The PRD's own case: "flip of corn" landed on revert_skill at 0.43."""
    orch = make_orch(FakeReasoner("flip a coin"), answers=["yes"])
    asyncio.run(orch.handle("revert_skill", "flip of corn", confidence=0.43))
    assert confirms(orch) == ["<did_you_mean flip a coin>"]
    assert [label for label, _ in orch.registry.calls] == ["flip_a_coin"]
    assert "<unknown>" not in orch._persona.spoken


def test_a_correction_may_land_on_an_inline_action():
    orch = make_orch(FakeReasoner("go to sleep"), answers=["yes"])
    asyncio.run(orch.handle("unknown", "go to sheep"))
    assert orch.standby is True
    assert "<standby>" in orch._persona.spoken


# -- declined, or nothing plausible -----------------------------------------------

def test_a_declined_correction_falls_back_to_the_plain_line():
    reasoner = FakeReasoner("flip a coin")
    orch = make_orch(reasoner, answers=["no"])
    asyncio.run(orch.handle("unknown", "flip a corn"))

    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<did_you_mean flip a coin>", "<unknown>"]
    assert len(reasoner.calls) == 1  # one guess, one confirmation — no second try


@pytest.mark.parametrize("reply", ["none", "None.", "NONE", '"none"', "", "   \n"])
def test_none_from_the_reasoner_asks_nothing(reply):
    orch = make_orch(FakeReasoner(reply), answers=["yes"])
    asyncio.run(orch.handle("unknown", "flibber"))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_a_failing_reasoner_falls_back_to_the_plain_line():
    orch = make_orch(FakeReasoner(TimeoutError("ollama timed out")), answers=["yes"])
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_a_guess_the_nlu_does_not_know_either_is_not_offered():
    """Asking "did you mean X?" and then failing on X would be worse than
    just saying it wasn't caught."""
    orch = make_orch(FakeReasoner("flip a corn cob"), answers=["yes"])
    asyncio.run(orch.handle("unknown", "flip a corn"))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_a_guess_that_is_itself_an_unsure_meta_action_is_not_offered():
    orch = make_orch(FakeReasoner("revert the coin skill"), answers=["yes"])
    asyncio.run(orch.handle("unknown", "revert the corn skill"))
    assert orch._persona.spoken == ["<unknown>"]


def test_a_guess_that_repeats_what_was_heard_is_not_offered():
    orch = make_orch(FakeReasoner("Revert the coin skill."), answers=["yes"])
    asyncio.run(orch.handle("revert_skill", "revert the coin skill", confidence=0.45))
    assert orch._persona.spoken == ["<unknown>"]


def test_a_guess_that_sounds_nothing_like_what_was_heard_is_not_offered():
    """A mishearing keeps most of the sounds. A small model that would rather
    answer than say "none" must not turn every stray sentence into a
    question."""
    orch = make_orch(FakeReasoner("flip a coin"), answers=["yes"])
    asyncio.run(orch.handle("unknown", "what is the capital of france"))
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


# -- the prompt --------------------------------------------------------------------

def test_the_prompt_carries_what_was_heard_and_what_jarvis_knows():
    reasoner = FakeReasoner("none")
    orch = make_orch(reasoner)
    asyncio.run(orch.handle("unknown", "flip a corn"))

    (system, prompt, options) = reasoner.calls[0]
    assert "flip a corn" in prompt
    # every registered skill: its name and some of its examples
    assert "flip_a_coin" in prompt
    assert "heads or tails?" in prompt
    # the seed intents' patterns
    for tag in ("goodbye", "shutdown", "teach"):
        assert intent_meta()[tag].patterns[-1].replace("{intent}", "").strip() in prompt
    assert "search:" in prompt
    assert "none" in system.lower()
    # a guess, not prose: no sampling, and it must not hold a turn for long
    assert options.get("temperature") == 0.0
    assert options.get("timeout") == orch.config.reasoner.guess_timeout_s


def test_vocabulary_lists_a_few_examples_per_command():
    many = SkillManifest(
        name="timer", description="x", examples=[f"set a timer {i}" for i in range(20)]
    )
    lines = mishear.vocabulary([many], intent_meta())
    timer = next(line for line in lines if line.startswith("timer:"))
    assert timer.count("set a timer") == mishear.EXAMPLES_PER_COMMAND
    # a seed pattern's {intent} slot is shown as a slot, not as the literal
    assert not any("{intent}" in line for line in lines)
    assert any(line.startswith("search:") for line in lines)


@pytest.mark.parametrize(
    "reply, heard, expected",
    [
        ("flip a coin", "flip a corn", "flip a coin"),
        ('"Flip a coin."', "flip a corn", "Flip a coin"),
        ("“flip a coin”", "flip a corn", "flip a coin"),
        ("Intended command: flip a coin", "flip a corn", "flip a coin"),
        ("flip a coin\nBecause corn sounds like coin.", "flip a corn", "flip a coin"),
        ("none", "flip a corn", None),
        ("None - nothing fits", "flip a corn", None),
        ("N/A", "flip a corn", None),
        ("", "flip a corn", None),
        ("Flip a corn!", "flip a corn", None),  # no correction at all
        # an explanation where a command should be
        ("I think the speaker most likely wanted to flip a coin of some kind here",
         "flip a corn", None),
    ],
)
def test_parse_guess(reply, heard, expected):
    assert mishear.parse_guess(reply, heard) == expected


# -- the whole loop ----------------------------------------------------------------

def test_run_loop_misheard_then_yes_dispatches(capsys):
    orch = make_orch(FakeReasoner("flip a coin"))
    orch.standby = True
    orch.mic = FakeMic(script=[True, True, False])
    orch.stt = ScriptedSTT(["flip a corn", "yes"])
    asyncio.run(orch.run())

    assert [label for label, _ in orch.registry.calls] == ["flip_a_coin"]
    out = capsys.readouterr().out
    assert 'heard   : "flip a corn"' in out
    assert 'guess   : "flip a coin"' in out  # why it asked is visible, as "heard" is
    assert orch.state == "idle"


def test_a_cancel_instead_of_an_answer_drops_the_turn():
    """Enter (or the edge's button) during "did you mean" abandons the turn:
    nothing is dispatched, and the loop carries on."""
    orch = make_orch(FakeReasoner("flip a coin"))
    orch.standby = True
    orch.stt = ScriptedSTT(["flip a corn", "yes"])

    class CancelledAtTheQuestion(FakeMic):
        def record_utterance(self, *a, **k):
            if self._i == 1:  # the capture that would hear the answer
                orch._interrupter.cancel_flag.set()
            return super().record_utterance(*a, **k)

    orch.mic = CancelledAtTheQuestion(script=[True, False, False])
    asyncio.run(orch.run())

    assert confirms(orch) == ["<did_you_mean flip a coin>"]
    assert orch.registry.calls == []
    assert "<unknown>" not in orch._persona.spoken
    assert "<standby>" in orch._persona.spoken  # the session ended the normal way


# -- the Ollama client ---------------------------------------------------------------

class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": " flip a coin \n"}


def test_generate_takes_a_temperature_and_a_timeout(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("jarvis.core.reasoner.requests.post", fake_post)
    r = Reasoner(base_url="http://ollama.test", model="m", timeout=30.0)

    assert r.generate("sys", "prompt", temperature=0.0, timeout=4.0) == "flip a coin"
    assert sent["json"]["options"]["temperature"] == 0.0
    assert sent["timeout"] == 4.0

    # the persona's rewrite call is unchanged
    r.generate("sys", "prompt")
    assert sent["json"]["options"]["temperature"] == 0.7
    assert sent["timeout"] == 30.0


def test_every_persona_has_a_did_you_mean_line(config):
    from jarvis.core.persona import PERSONAS_DIR, Persona

    for root in sorted(p for p in PERSONAS_DIR.iterdir() if p.is_dir()):
        line = Persona.load(root.name, config).line("did_you_mean")
        assert "{guess}" in line, f"persona {root.name!r}: {line!r}"
