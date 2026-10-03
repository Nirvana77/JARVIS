"""From the owner's first session on the dev brain after M7 (2026-10-03).

- "What is today's date?" -> recall_memory: "You have not asked me to
  remember anything." There was no date or time skill at all.
- Both machines run on UTC, the owner lives two hours ahead: the time must
  come from ``[general] timezone``, for the clock and the reasoner's "Now".
- "Fetch today's modes from the watch." -> a plan of the same command twice.
- "Exactly." to "Did you mean …?" was not a yes.
- "Learn how to get the current date." -> "I'm afraid I didn't catch that":
  learning was off on the dev brain, and it said so as if it had not heard.
- Two brains on one ``data/`` must not both retrain on the same phrasings.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
from pathlib import Path

import pytest

from jarvis.config import GeneralConfig, LearningConfig, load_config
from jarvis.core.context import Context
from jarvis.factory.flows import ask_yes_no_or_none
from jarvis.learning import Learning
from jarvis.skills.builtin import clock
from tests.test_misheard import FakeReasoner

NOON_UTC = dt.datetime(2026, 10, 3, 17, 22, tzinfo=dt.timezone.utc)


def _ctx(config, tz="Europe/Stockholm"):
    cfg = dataclasses.replace(config, general=GeneralConfig(language="en", timezone=tz))
    return Context(say=lambda s: None, config=cfg, _data_dir=Path("/nonexistent"),
                   clock=lambda: NOON_UTC)


# -- the clock ---------------------------------------------------------------------------

def test_the_time_is_in_the_configured_zone(config):
    assert _ctx(config).now() == NOON_UTC.astimezone(dt.timezone(dt.timedelta(hours=2)))
    assert _ctx(config).now().hour == 19


def test_without_a_zone_the_machines_own_is_used(config):
    assert _ctx(config, tz="").now().tzinfo is not None


def test_a_misspelt_zone_falls_back_rather_than_failing(tmp_path):
    (tmp_path / "config.toml").write_text('[general]\ntimezone = "Mars/Olympus"\n', encoding="utf-8")
    assert load_config(tmp_path / "config.toml").general.timezone == ""


def test_the_clock_says_time_and_date_in_voice(config):
    line = clock.run(_ctx(config))
    assert line == "It's 19:22 on Saturday the 3rd of October, sir."
    assert clock.MANIFEST.voice == "jarvis"


def test_ordinals():
    assert [clock.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "31st"
    ]


def test_the_reasoners_now_is_in_the_configured_zone(config):
    from tests.test_reasoning import make_orch

    cfg = dataclasses.replace(config, general=GeneralConfig(language="en", timezone="Europe/Stockholm"))
    o = make_orch(config=cfg)
    o._clock_now = lambda: NOON_UTC
    assert o._now().hour == 19


# -- plans and yes ------------------------------------------------------------------------

def test_a_plan_that_repeats_a_command_runs_it_once(config):
    from tests.test_reasoning import make_orch

    reply = '{"commands": ["flip a coin", "flip a coin please"]}'
    o = make_orch(reasoner=FakeReasoner(reply), answers=["yes"])
    o.nlu.mapping["flip a coin please"] = o.nlu.mapping["flip a coin"]
    o.config = dataclasses.replace(
        o.config, reasoner=dataclasses.replace(o.config.reasoner, correct_misheard=False)
    )
    asyncio.run(o.handle("unknown", "toss me something", 0.1))
    assert [label for label, _ in o.registry.calls] == ["flip_a_coin"]
    assert any("flip a coin'" in s or "flip a coin" in s for s in o._persona.spoken)
    assert not any("then" in s for s in o._persona.spoken)


@pytest.mark.parametrize("reply, meant", [
    ("Exactly.", True), ("That's right.", True), ("Precisely", True), ("Spot on", True),
    ("yes please", True), ("not exactly", None), ("that's not right", None),
])
def test_more_ways_of_saying_yes(reply, meant):
    async def ask(prompt):
        return reply
    assert asyncio.run(ask_yes_no_or_none(ask, "Did you mean it?")) is meant


# -- learning on a brain where it is off ----------------------------------------------------

def test_with_learning_off_jarvis_says_it_cannot_learn_here(config):
    from tests.test_reasoning import make_orch

    cfg = dataclasses.replace(
        config,
        learning=dataclasses.replace(LearningConfig(), enabled=False),
        reasoner=dataclasses.replace(config.reasoner, correct_misheard=False),
    )
    o = make_orch(reasoner=FakeReasoner('{"learn": "get the current date"}'), config=cfg)
    o.learning = Learning.in_memory(cfg)
    o.claude_client = type("C", (), {"available": True})()
    o.sandbox = object()
    asyncio.run(o.handle("unknown", "learn how to get the current date", 0.1))
    assert o._persona.spoken[-1] == "<learning_off>"


# -- two brains, one data/ ------------------------------------------------------------------

def test_a_brain_retrains_only_on_what_it_learned_itself(config):
    from tests.test_reasoning import make_orch

    cfg = dataclasses.replace(config, learning=dataclasses.replace(LearningConfig(), retrain_after=1))
    o = make_orch(config=cfg)
    o.learning = Learning.in_memory(cfg)
    # the other brain learned this one; it is in the shared file, pending
    o.learning.phrasings.add("toss one", "flip_a_coin", why="unsure")
    assert not o._phrasings_due()
    o._learn_phrasing("chuck a coin", "flip_a_coin", "confirmed")
    assert o._phrasings_due()
    assert [p.text for p in o._own_pending()] == ["chuck a coin"]


# -- asking again means the answer was wrong (the owner, 2026-10-03) -------------------------
#
# "What is today's date?" -> recall_memory: "You have not asked me to remember
# anything." / "What is the date of today?" -> the same / "No. What is the
# current date?" -> the same. "The brain should understand that I meant
# something else and try to solve it. I asked the same question two times but
# JARVIS did not learn."

from jarvis.config import LearningConfig as _LC  # noqa: E402
from jarvis.nlu.classifier import Prediction  # noqa: E402
from jarvis.skills.contract import SkillManifest  # noqa: E402
from tests.test_orchestrator import FakeMic, FakeRegistry, FakeTTS, FakeWake  # noqa: E402
from tests.test_misheard import DidYouMeanPersona, ScriptedSTT  # noqa: E402

DATE_Q1 = "what is today's date?"
DATE_Q2 = "what is the date of today?"
DATE_Q3 = "no. what is the current date?"


class RankedNLU:
    def __init__(self, mapping):
        self.mapping = mapping

    def explain(self, text):
        label, conf, ranking = self.mapping.get(text.lower(), ("unknown", 0.1, [("unknown", 0.1)]))
        return Prediction(label, conf, ranking, 0.9)

    def predict(self, text):
        p = self.explain(text)
        return p.label, p.confidence


MAPPING = {
    DATE_Q1: ("recall_memory", 0.54, [("recall_memory", 0.54), ("search", 0.14)]),
    DATE_Q2: ("recall_memory", 0.42, [("recall_memory", 0.42), ("search", 0.18)]),
    DATE_Q3: ("correction", 0.41, [("correction", 0.41), ("search", 0.18)]),
    "what is the current date?": ("recall_memory", 0.40, [("recall_memory", 0.40)]),
    "flip a coin": ("flip_a_coin", 0.9, [("flip_a_coin", 0.9)]),
    "what time is it": ("clock", 0.9, [("clock", 0.9)]),
}

COIN = SkillManifest(name="flip_a_coin", description="Flip a coin.", examples=["flip a coin"], origin="learned")
CLOCK = SkillManifest(name="clock", description="Tell the time.", examples=["what time is it"])


class ClaudeUp:
    available = True


def make(answers=(), reasoner=None, **learning):
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.nlu.corpus import intent_meta

    config = load_config()
    config = dataclasses.replace(
        config,
        learning=dataclasses.replace(_LC(), **learning),
        reasoner=dataclasses.replace(config.reasoner, correct_misheard=False),
    )
    persona = DidYouMeanPersona()
    o = Orchestrator(
        config=config, wake=FakeWake(), mic=FakeMic(script=[True] * len(answers)),
        stt=ScriptedSTT(answers), tts=FakeTTS(persona), nlu=RankedNLU(MAPPING),
        persona=persona, registry=FakeRegistry(manifests=[COIN, CLOCK]),
        intent_meta=intent_meta(), reasoner=reasoner, learning=Learning.in_memory(config),
    )
    o._persona = persona
    o.standby = False
    return o


def turn(o, text):
    label, conf = o.nlu.predict(text)
    asyncio.run(o._turn(label, text, conf))


class Recording(FakeReasoner):
    pass


def test_a_question_asked_again_is_not_answered_the_same_way():
    o = make(reasoner=Recording('{"answer": "It is Saturday the 3rd of October, sir."}'))
    turn(o, DATE_Q1)
    assert o._persona.spoken[-1] == "<nothing_remembered>"
    turn(o, DATE_Q2)
    assert o._persona.spoken[-1] == "It is Saturday the 3rd of October, sir."
    assert o.learning.log.records()[-1]["path"] == "retry"


def test_the_reasoner_is_told_the_last_answer_did_not_help():
    reasoner = Recording('{"answer": "It is Saturday, sir."}')
    o = make(reasoner=reasoner)
    turn(o, DATE_Q1)
    turn(o, DATE_Q2)
    _system, prompt, _opts = reasoner.calls[-1]
    assert "asking again" in prompt
    assert "<nothing_remembered>" in prompt  # what was said, so it is not said again


def test_a_no_in_front_of_the_repeat_is_a_retry_too():
    o = make(reasoner=Recording('{"answer": "It is Saturday, sir."}'))
    turn(o, DATE_Q1)
    turn(o, DATE_Q3)
    assert o._persona.spoken[-1] == "It is Saturday, sir."
    assert o.learning.log.records()[-1]["path"] == "retry"


def test_a_command_said_twice_is_done_twice():
    o = make(reasoner=Recording('{"none": true}'))
    turn(o, "flip a coin")
    turn(o, "flip a coin")
    assert [c[0] for c in o.registry.calls] == ["flip_a_coin", "flip_a_coin"]


def test_a_retry_that_finds_a_skill_learns_both_phrasings():
    plan = Recording('{"commands": ["what time is it"]}')
    o = make(answers=["yes"], reasoner=plan)
    turn(o, DATE_Q1)
    turn(o, DATE_Q2)
    assert o.registry.calls[-1][0] == "clock"
    learned = set(o.learning.phrasings.examples())
    assert {(DATE_Q1, "clock"), (DATE_Q2, "clock")} <= learned


def test_a_retry_nothing_can_answer_is_learned_as_a_new_skill():
    o = make(reasoner=Recording('{"none": true}'))
    o.claude_client, o.sandbox = ClaudeUp(), object()
    started = []

    async def start(request, announce=True):
        started.append(request)

    o._start_learning = start
    turn(o, DATE_Q1)
    turn(o, DATE_Q2)
    (request,) = started
    assert request.autonomous and request.versioning == "new"
    assert DATE_Q1 in request.spec.examples and DATE_Q2 in request.spec.examples


def test_after_the_window_a_question_is_just_a_question():
    o = make(reasoner=Recording('{"answer": "It is Saturday, sir."}'), correction_window_s=0.0)
    turn(o, DATE_Q1)
    o._previous["at"] -= 1.0
    turn(o, DATE_Q2)
    assert o._persona.spoken[-1] == "<nothing_remembered>"


def test_the_2024_tutorial_intents_are_gone():
    """"The weather is nice" and "We accept VISA, Mastercard and AMEX" were
    made-up answers from the 2024 tutorial chatbot. Without them a weather
    question reaches the reasoner, and can be learned (owner, 2026-10-03)."""
    from jarvis.nlu.corpus import intent_meta

    assert not {"weather", "hours", "payments", "opentoday"} & set(intent_meta())
