"""M7 parts B–D and G in the orchestrator: every turn is logged, confirmed
turns teach the classifier the words that were actually heard, a correction
undoes that, and "what have you learned today?" says what was learned.

Fakes as in ``test_misheard.py``; the learning stores are in RAM
(``Learning.in_memory``), so nothing touches ``data/``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

from jarvis.config import LearningConfig, load_config
from jarvis.core.orchestrator import Orchestrator
from jarvis.learning import Learning
from jarvis.nlu.classifier import Prediction
from jarvis.nlu.corpus import intent_meta
from jarvis.skills.contract import SkillManifest
from tests.test_misheard import DidYouMeanPersona, FakeReasoner, ScriptedSTT
from tests.test_orchestrator import FakeMic, FakeRegistry, FakeTTS, FakeWake

COIN = SkillManifest(
    name="flip_a_coin",
    description="Flip one or more coins and report heads or tails.",
    examples=["flip a coin", "heads or tails?"],
    origin="learned",
)
TIMER = SkillManifest(
    name="set_timer",
    description="Set a timer.",
    examples=["set a timer for five minutes"],
    origin="builtin",
)


class RankedNLU:
    """A fake classifier with a full ranking, so the offline "did you mean"
    has a runner-up to offer. ``mapping`` text -> (label, conf, ranking)."""

    def __init__(self, mapping, version=7):
        self.mapping = mapping
        self.version = version
        self.seen = []

    def explain(self, text):
        self.seen.append(text)
        label, conf, ranking = self.mapping.get(
            text.lower(), ("unknown", 0.1, [("unknown", 0.1)])
        )
        return Prediction(label, conf, ranking, 0.9)

    def predict(self, text):
        p = self.explain(text)
        return p.label, p.confidence


MAPPING = {
    "flip a coin": ("flip_a_coin", 0.9, [("flip_a_coin", 0.9)]),
    "toss one": ("flip_a_coin", 0.42, [("flip_a_coin", 0.42), ("set_timer", 0.2)]),
    # known issue #18: a removal that ran the skill it meant to remove
    "forget how to flip a coin": (
        "flip_a_coin", 0.43, [("flip_a_coin", 0.43), ("remove_skill", 0.22)]
    ),
    # unknown, but close: the offline confirm offers flip_a_coin
    "chuck a coin": ("unknown", 0.3, [("flip_a_coin", 0.3), ("set_timer", 0.1)]),
    # unknown and nowhere near
    "flibber": ("unknown", 0.1, [("flip_a_coin", 0.1)]),
    "set a timer for five minutes": ("set_timer", 0.9, [("set_timer", 0.9)]),
    "no i meant set a timer for five minutes": ("correction", 0.9, [("correction", 0.9)]),
    "that's wrong": ("correction", 0.9, [("correction", 0.9)]),
    "what have you learned today": ("learned_today", 0.9, [("learned_today", 0.9)]),
    "revert the coin skill": ("revert_skill", 0.9, [("revert_skill", 0.9)]),
}


def make(answers=(), reasoner=None, mapping=MAPPING, **learning):
    config = load_config()
    config = dataclasses.replace(
        config, learning=dataclasses.replace(LearningConfig(), **learning)
    )
    persona = DidYouMeanPersona()
    o = Orchestrator(
        config=config,
        wake=FakeWake(),
        mic=FakeMic(script=[True] * len(answers)),
        stt=ScriptedSTT(answers),
        tts=FakeTTS(persona),
        nlu=RankedNLU(mapping),
        persona=persona,
        registry=FakeRegistry(manifests=[COIN, TIMER]),
        intent_meta=intent_meta(),
        reasoner=reasoner,
        learning=Learning.in_memory(config),
    )
    o._persona = persona
    o.wake.stop = o.stop
    o.standby = False
    return o


def turn(o, text):
    label, conf = o.nlu.predict(text)
    asyncio.run(o._turn(label, text, conf))


def records(o):
    return o.learning.log.records()


def learned(o):
    return o.learning.phrasings.examples()


# -- B: one log line per turn -------------------------------------------------------

def test_a_direct_skill_turn_is_logged():
    o = make()
    turn(o, "flip a coin")
    (r,) = records(o)
    assert r["heard"] == "flip a coin" and r["label"] == "flip_a_coin"
    assert r["path"] == "direct" and r["skill"] == "flip_a_coin" and r["outcome"] == "ok"
    assert r["device"] == "local" and r["said"]
    assert r["model"] == 7


def test_an_unknown_turn_is_logged_as_unknown():
    o = make()
    turn(o, "flibber")
    (r,) = records(o)
    assert r["path"] == "unknown" and "skill" not in r


def test_a_failing_skill_is_logged_as_an_error():
    o = make()
    o.registry.raises = RuntimeError("boom")
    turn(o, "flip a coin")
    (r,) = records(o)
    assert r["outcome"] == "error" and "boom" in r["error"]


def test_a_meta_action_is_logged_by_its_action():
    o = make()
    turn(o, "what have you learned today")
    assert records(o)[0]["path"] == "learned_today"


def test_with_the_log_off_nothing_is_logged():
    o = make(log=False)
    turn(o, "flip a coin")
    assert records(o) == []


def test_forget_everything_also_forgets_the_log():
    o = make(answers=["yes"])
    turn(o, "flip a coin")
    o.memory.device().remember("the gate code is 4821")
    asyncio.run(o._turn("forget_memory", "forget everything", 0.9))
    assert records(o) == []  # the earlier turn went, and the forget turn is not kept


# -- C: confirmed turns teach the words that were heard -------------------------------

def test_a_yes_to_a_mishearing_learns_the_heard_words():
    o = make(answers=["yes"], reasoner=FakeReasoner("flip a coin"))
    o.nlu.mapping["flip a corn"] = ("unknown", 0.2, [("unknown", 0.2)])
    turn(o, "flip a corn")
    assert learned(o) == [("flip a corn", "flip_a_coin")]
    (r,) = records(o)
    assert r["path"] == "mishear" and r["skill"] == "flip_a_coin" and r["learned"]


def test_a_no_to_a_mishearing_learns_nothing():
    o = make(answers=["no"], reasoner=FakeReasoner("flip a coin"))
    o.nlu.mapping["flip a corn"] = ("unknown", 0.2, [("unknown", 0.2)])
    turn(o, "flip a corn")
    assert learned(o) == []
    assert records(o)[0]["outcome"] == "declined"


def test_a_confirmed_one_step_plan_learns_the_heard_words():
    plan = FakeReasoner('{"commands": ["flip a coin"]}')
    o = make(answers=["yes"], reasoner=plan)
    o.config = dataclasses.replace(
        o.config, reasoner=dataclasses.replace(o.config.reasoner, correct_misheard=False)
    )
    turn(o, "give me a random heads or tails")
    assert learned(o) == [("give me a random heads or tails", "flip_a_coin")]
    assert records(o)[0]["path"] == "plan"


def test_a_confirmed_plan_of_several_steps_learns_nothing():
    plan = FakeReasoner('{"commands": ["flip a coin", "set a timer for five minutes"]}')
    o = make(answers=["yes"], reasoner=plan)
    o.config = dataclasses.replace(
        o.config, reasoner=dataclasses.replace(o.config.reasoner, correct_misheard=False)
    )
    turn(o, "coin then timer please")
    assert learned(o) == []


def test_without_a_reasoner_a_near_miss_is_offered_and_learned():
    o = make(answers=["yes"])
    turn(o, "chuck a coin")
    assert o._persona.spoken[0] == "<did_you_mean flip a coin>"
    assert o.registry.calls and o.registry.calls[0][0] == "flip_a_coin"
    assert learned(o) == [("chuck a coin", "flip_a_coin")]
    assert records(o)[0]["path"] == "confirmed"


def test_a_far_miss_is_not_offered():
    o = make(answers=["yes"])
    turn(o, "flibber")
    assert o._persona.spoken == ["<unknown>"]
    assert learned(o) == []


def test_confirm_margin_zero_turns_the_offline_offer_off():
    o = make(answers=["yes"], confirm_margin=0.0)
    turn(o, "chuck a coin")
    assert o._persona.spoken == ["<unknown>"]


def test_an_unsure_turn_nobody_corrected_is_learned_at_the_next_turn():
    o = make()
    turn(o, "toss one")          # flip_a_coin at 0.42: ran, but unsure
    assert learned(o) == []      # not yet: a correction may still come
    turn(o, "flip a coin")
    assert learned(o) == [("toss one", "flip_a_coin")]


def test_an_unsure_turn_whose_runner_up_is_a_meta_action_is_not_learned():
    """Known issue #18: "forget how to flip a coin" runs flip_a_coin at 0.43
    with remove_skill second. Learning it would make that mistake permanent."""
    o = make()
    o._last_prediction = ("forget how to flip a coin", o.nlu.explain("forget how to flip a coin"))
    turn(o, "forget how to flip a coin")
    turn(o, "flip a coin")
    assert learned(o) == []


def test_an_unsure_turn_is_learned_when_the_session_ends():
    o = make()
    turn(o, "toss one")
    o._commit_provisional()
    assert learned(o) == [("toss one", "flip_a_coin")]


def test_a_sure_turn_teaches_nothing():
    o = make()
    turn(o, "flip a coin")
    turn(o, "set a timer for five minutes")
    assert learned(o) == []


def test_learning_off_logs_but_learns_nothing():
    o = make(answers=["yes"], enabled=False)
    turn(o, "chuck a coin")
    assert learned(o) == []
    assert records(o)[0]["path"] == "confirmed"


def test_a_guarded_action_is_never_learned():
    o = make()
    assert o._learn_phrasing("undo that coin thing", "revert_skill", "mishear") is None
    assert o._learn_phrasing("note that milk", "note", "mishear") is None
    assert o._learn_phrasing("x", "unknown", "mishear") is None
    assert learned(o) == []


# -- C: retraining on what was learned -------------------------------------------------

class FakeClassifier:
    def __init__(self, mapping):
        self.mapping = mapping
        self.closed = False

    def predict(self, text):
        return self.mapping.get(text, ("unknown", 0.1))

    def close(self):
        self.closed = True


def trainer(new_mapping, calls):
    async def train_and_load(examples):
        calls.append(examples)
        return "ok", (SimpleNamespace(path="/nonexistent/v99"), FakeClassifier(new_mapping))
    return train_and_load


def test_enough_phrasings_trigger_a_retrain_that_is_staged_and_announced():
    calls = []
    o = make(retrain_after=2)
    good = {"search black holes": ("search", 0.9), "open github": ("open_app", 0.9),
            "go to sleep": ("goodbye", 0.9), "toss one": ("flip_a_coin", 0.9)}
    o._train_and_load = trainer(good, calls)
    o.nlu = FakeClassifier(good)  # the live model already gets the probes right

    async def go():
        o._learn_phrasing("toss one", "flip_a_coin", "unsure")
        assert not o._phrasings_due()
        o._learn_phrasing("chuck a coin", "flip_a_coin", "confirmed")
        assert o._phrasings_due()
        await o._idle_tick()
        await o._phrasing_task
    asyncio.run(go())

    assert calls, "no retrain"
    sources = {(e.text, e.source) for e in calls[0]}
    assert ("toss one", "learned") in sources and ("chuck a coin", "learned") in sources
    assert o._staged is not None
    assert o.learning.phrasings.pending() == []
    assert o._pending_announcement == "<learned_phrasings>"


def test_a_retrain_that_gets_worse_is_discarded_and_its_phrasings_quarantined():
    calls = []
    o = make(retrain_after=1)
    old = {"search black holes": ("search", 0.9), "open github": ("open_app", 0.9),
           "go to sleep": ("goodbye", 0.9)}
    worse = dict(old, **{"go to sleep": ("flip_a_coin", 0.9)})
    o.nlu = FakeClassifier(old)
    o._train_and_load = trainer(worse, calls)

    async def go():
        o._learn_phrasing("go to bed coin", "flip_a_coin", "unsure")
        await o._idle_tick()
        await o._phrasing_task
    asyncio.run(go())

    assert o._staged is None
    assert learned(o) == []
    assert o.learning.phrasings.is_rejected("go to bed coin", "flip_a_coin")


def test_recent_good_turns_are_regression_probes():
    o = make(retrain_after=1)
    turn(o, "set a timer for five minutes")  # logged: direct, ok, sure
    old = {"search black holes": ("search", 0.9), "open github": ("open_app", 0.9),
           "go to sleep": ("goodbye", 0.9), "set a timer for five minutes": ("set_timer", 0.9)}
    worse = dict(old, **{"set a timer for five minutes": ("flip_a_coin", 0.9)})
    o.nlu = FakeClassifier(old)
    o._train_and_load = trainer(worse, [])

    async def go():
        o._learn_phrasing("timer coin", "flip_a_coin", "unsure")
        await o._idle_tick()
        await o._phrasing_task
    asyncio.run(go())
    assert o._staged is None


def test_no_retrain_while_a_learning_job_runs():
    o = make(retrain_after=1)
    o._learn_phrasing("toss one", "flip_a_coin", "unsure")
    o._jobs["something"] = object()
    assert not o._phrasings_due()


def test_every_retrain_trains_on_the_learned_phrasings_too():
    """A learning job's retrain, an edge's new tools and the phrasing
    retrain all build their corpus through ``_corpus``: one that left the
    phrasings out would quietly unlearn them at the next swap."""
    o = make()
    o._learn_phrasing("toss one", "flip_a_coin", "unsure")
    corpus = o._corpus(o.registry.manifests())
    assert ("toss one", "flip_a_coin", "learned") in {(e.text, e.label, e.source) for e in corpus}


# -- D: corrections ----------------------------------------------------------------------

def test_a_correction_undoes_what_was_learned_and_learns_the_right_label():
    o = make(answers=["yes"])
    turn(o, "chuck a coin")                       # offered, confirmed, learned
    assert learned(o) == [("chuck a coin", "flip_a_coin")]
    turn(o, "no I meant set a timer for five minutes")
    assert learned(o) == [("chuck a coin", "set_timer")]
    assert o.learning.phrasings.is_rejected("chuck a coin", "flip_a_coin")
    assert o.registry.calls[-1][0] == "set_timer"
    assert o.learning.state.confusions() == {"flip_a_coin -> set_timer": 1}
    assert records(o)[-1]["path"] == "correction"
    assert records(o)[-1]["corrects"] == records(o)[0]["id"]


def test_a_correction_drops_an_unsure_turn_before_it_is_learned():
    o = make(answers=["set a timer for five minutes"])
    turn(o, "toss one")
    turn(o, "that's wrong")                       # no "I meant": asked
    assert "<what_did_you_mean>" in o._persona.spoken
    assert learned(o) == [("toss one", "set_timer")]
    assert o.registry.calls[-1][0] == "set_timer"


def test_a_correction_with_nothing_to_correct_says_so():
    o = make()
    turn(o, "that's wrong")
    assert o._persona.spoken[-1] == "<nothing_to_correct>"
    assert learned(o) == []


def test_a_correction_after_the_window_corrects_nothing():
    o = make(answers=["yes"], correction_window_s=0.0)
    turn(o, "chuck a coin")
    o._last_skill["at"] -= 1.0
    turn(o, "no I meant set a timer for five minutes")
    assert o._persona.spoken[-1] == "<nothing_to_correct>"
    assert learned(o) == [("chuck a coin", "flip_a_coin")]


# -- G: what was learned today ------------------------------------------------------------

def test_what_have_you_learned_today():
    o = make(answers=["yes"])
    turn(o, "what have you learned today")
    assert o._persona.spoken[-1] == "<nothing_learned>"
    turn(o, "chuck a coin")
    turn(o, "what have you learned today")
    line = o._persona.spoken[-1]
    assert "chuck a coin" in line
