"""M4.5 — reasoning and per-device memory, at the orchestrator
(`PRD/milestone-4.5-reasoning-and-memory.md`, "Verification").

Compound commands need no model. Planning and answering use the local
reasoner — here a fake with canned replies, one for M4's mishearing prompt and
one for the reasoning prompt, so nothing needs Ollama.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from jarvis.config import load_config
from jarvis.core import mishear, reasoning
from jarvis.core.memory import Memory
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.reasoner import Reasoner
from jarvis.nlu import compound
from jarvis.nlu.classifier import Prediction
from jarvis.nlu.corpus import intent_meta
from jarvis.skills.contract import SkillManifest
from tests.test_misheard import DidYouMeanPersona, ScriptedSTT
from tests.test_orchestrator import FakeMic, FakeRegistry, FakeTTS, FakeWake


# -- fakes ------------------------------------------------------------------

class ThinkingReasoner:
    """``guess`` answers M4's mishearing prompt, ``thought`` the reasoning
    prompt. Either may be an exception to raise."""

    def __init__(self, thought='{"none": true}', guess="none", available=True):
        self.thought = thought
        self.guess = guess
        self.available = available
        self.calls = []  # ("guess" | "thought", system, prompt, options)

    def generate(self, system, prompt, **options):
        kind = "guess" if system == mishear.SYSTEM else "thought"
        self.calls.append((kind, system, prompt, options))
        reply = self.guess if kind == "guess" else self.thought
        if isinstance(reply, Exception):
            raise reply
        return reply

    def kinds(self):
        return [c[0] for c in self.calls]


class SimilarityNLU:
    """``mapping`` values are ``(label, confidence)`` or ``(label, confidence,
    similarity)``. Case is ignored, as by the real classifier."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.seen = []

    def _lookup(self, text):
        self.seen.append(text)
        found = self.mapping.get(text.lower().strip(" .?!"), ("unknown", 0.1, 0.2))
        return found if len(found) == 3 else (*found, 1.0)

    def predict(self, text):
        label, conf, _sim = self._lookup(text)
        return label, conf

    def explain(self, text):
        label, conf, sim = self._lookup(text)
        return Prediction(label, conf, [(label, conf)], sim)


class Persona(DidYouMeanPersona):
    def __init__(self):
        super().__init__()
        self.phrased = []

    def phrase(self, text):
        self.phrased.append(text)
        return text

    def character(self):
        return "You are J.A.R.V.I.S., dry and precise."


MANIFESTS = [
    SkillManifest(name="search", description="Search.", examples=["search black holes"]),
    SkillManifest(name="note", description="Note.", examples=["note that the wifi code is 1234"]),
    SkillManifest(name="flip_a_coin", description="Flip.", examples=["flip a coin"], origin="learned"),
    SkillManifest(name="find_watch", description="Ring the watch.", examples=["find my watch"], origin="edge"),
    SkillManifest(
        name="set_timer", description="Set a timer.", examples=["set a timer for 5 minutes"],
        params={"duration": {"type": "duration", "required": True}}, origin="edge",
    ),
]

NLU = {
    "search black holes": ("search", 0.95),
    "flip a coin": ("flip_a_coin", 0.92),
    "find my watch": ("find_watch", 0.86),
    "set a timer for five minutes": ("set_timer", 0.90),
    "set a timer for 5 minutes": ("set_timer", 0.92),
    "set a timer for ten minutes": ("set_timer", 0.95),
    "go to sleep": ("goodbye", 0.9),
    "note that buy milk": ("note", 0.88),
    "play simon": ("search", 0.9),
    "garfunkel": ("search", 0.75, 0.35),          # confident, but near nothing it was trained on
    "30 seconds": ("set_timer", 0.46, 0.53),
    "remove the timer skill": ("remove_skill", 0.9),
    "what do you remember": ("recall_memory", 0.9),
    "forget everything i told you": ("forget_memory", 0.9),
}


def make_orch(reasoner=None, answers=(), config=None, memory=None):
    persona = Persona()
    o = Orchestrator(
        config=config or load_config(),
        wake=FakeWake(),
        mic=FakeMic(script=[True] * len(answers)),
        stt=ScriptedSTT(answers),
        tts=FakeTTS(persona),
        nlu=SimilarityNLU(dict(NLU)),
        persona=persona,
        registry=FakeRegistry(manifests=MANIFESTS),
        intent_meta=intent_meta(),
        reasoner=reasoner,
        memory=memory,
    )
    o._persona = persona
    o.wake.stop = o.stop
    o.standby = False
    return o


def configured(**sections):
    """The local config with fields replaced: ``configured(nlu={"compound": False})``."""
    config = load_config()
    return dataclasses.replace(
        config,
        **{name: dataclasses.replace(getattr(config, name), **fields) for name, fields in sections.items()},
    )


def labels(orch):
    return [label for label, _params in orch.registry.calls]


def confirms(orch):
    return [s for s in orch._persona.spoken if s.startswith("<did_you_mean")]


def handle(orch, text, label="unknown", confidence=0.2):
    asyncio.run(orch.handle(label, text, confidence))


# -- compound commands: no model needed ---------------------------------------------

def test_two_commands_in_one_sentence_run_in_order():
    orch = make_orch(reasoner=None)
    handle(orch, "set a timer for five minutes and find my watch")
    assert orch.registry.calls == [("set_timer", {"duration": 300}), ("find_watch", {})]
    assert confirms(orch) == []                      # the speaker's own words: nothing to ask
    assert "<unknown>" not in orch._persona.spoken


def test_then_and_commas_split_too_and_each_clause_keeps_its_own_slots():
    orch = make_orch()
    handle(orch, "search black holes, then set a timer for ten minutes and then find my watch")
    assert orch.registry.calls == [
        ("search", {"query": "black holes"}),
        ("set_timer", {"duration": 600}),
        ("find_watch", {}),
    ]


def test_a_confidently_classified_whole_is_still_split_when_it_is_two_commands():
    """The classifier often picks one half and is sure of it."""
    orch = make_orch()
    handle(orch, "flip a coin and find my watch", label="find_watch", confidence=0.55)
    assert labels(orch) == ["flip_a_coin", "find_watch"]


def test_a_stray_half_does_not_make_two_commands():
    """"garfunkel" alone classifies confidently — but it is near nothing the
    model was trained on."""
    orch = make_orch()
    handle(orch, "play simon and garfunkel", label="search", confidence=0.9)
    assert orch.registry.calls == [("search", {"query": "play simon and garfunkel"})]


def test_the_same_skill_twice_is_one_command():
    orch = make_orch()
    handle(orch, "set a timer for 5 minutes and 30 seconds", label="set_timer", confidence=0.91)
    assert orch.registry.calls == [("set_timer", {"duration": 330})]


def test_dictation_is_never_split():
    orch = make_orch()
    handle(orch, "note that buy milk and find my watch", label="note", confidence=0.7)
    assert labels(orch) == ["note"]


def test_a_chain_never_contains_a_session_or_meta_action():
    orch = make_orch()
    handle(orch, "flip a coin and go to sleep", label="flip_a_coin", confidence=0.7)
    assert labels(orch) == ["flip_a_coin"]
    assert orch.standby is False


def test_compound_can_be_switched_off():
    orch = make_orch(config=configured(nlu={"compound": False}))
    handle(orch, "set a timer for five minutes and find my watch")
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]


def test_nothing_is_split_in_standby():
    orch = make_orch()
    orch.standby = True
    handle(orch, "flip a coin and find my watch", label="find_watch", confidence=0.55)
    assert orch.registry.calls == []


@pytest.mark.parametrize(
    "text, clauses",
    [
        ("set a timer for five minutes and find my watch",
         ["set a timer for five minutes", "find my watch"]),
        ("play some jazz, then set a timer", ["play some jazz", "set a timer"]),
        ("open github and then search cats and play lofi", ["open github", "search cats", "play lofi"]),
        ("Flip a coin. Then find my watch.", ["Flip a coin", "find my watch"]),
        ("search black holes", []),
        ("and", []),
        ("search for sand and", []),                      # nothing after the "and"
        ("a and b and c and d and e and f", []),          # more clauses than a command has
    ],
)
def test_split(text, clauses):
    assert compound.split(text) == clauses


# -- planning: the reasoner re-says the request as commands -----------------------------

PLAN = json.dumps({"commands": ["set a timer for 5 minutes", "find my watch"]})


def test_a_plan_is_classified_confirmed_once_and_run_in_order():
    reasoner = ThinkingReasoner(PLAN)
    orch = make_orch(reasoner, answers=["yes"])
    handle(orch, "wake me in five and beep the watch")

    assert confirms(orch) == ["<did_you_mean set a timer for 5 minutes, then find my watch>"]
    assert orch.registry.calls == [("set_timer", {"duration": 300}), ("find_watch", {})]
    assert "<unknown>" not in orch._persona.spoken
    assert reasoner.kinds() == ["guess", "thought"]   # the mishearing check came first


def test_a_declined_plan_runs_nothing():
    reasoner = ThinkingReasoner(PLAN)
    orch = make_orch(reasoner, answers=["no"])
    handle(orch, "wake me in five and beep the watch")
    assert orch.registry.calls == []
    assert orch._persona.spoken[-1] == "<unknown>"
    assert reasoner.kinds().count("thought") == 1     # no second plan


def test_an_unclear_reply_to_a_plan_runs_nothing():
    orch = make_orch(ThinkingReasoner(PLAN), answers=["I'm unsure"])
    handle(orch, "wake me in five and beep the watch")
    assert orch.registry.calls == []
    assert orch._persona.spoken[-1] == "<unknown>"


def test_an_interrupt_during_a_chain_stops_the_rest_of_it():
    """The edge's button (or Enter) while step one runs: step two must not."""
    orch = make_orch()
    orch.registry.on_dispatch = orch._interrupter.cancel_flag.set
    handle(orch, "search black holes and find my watch")
    assert labels(orch) == ["search"]


def test_an_interrupt_while_the_reasoner_thinks_drops_the_turn():
    """Nothing is asked and nothing is said once the speaker has cancelled."""
    orch = make_orch(answers=["yes"])

    class Interrupted(ThinkingReasoner):
        def generate(self, system, prompt, **options):
            orch._interrupter.cancel_flag.set()
            return super().generate(system, prompt, **options)

    orch.reasoner = Interrupted(PLAN)
    handle(orch, "wake me in five and beep the watch")
    assert orch._persona.spoken == []
    assert orch.registry.calls == []


def test_a_reasoner_that_failed_the_guess_is_not_asked_to_think_as_well():
    """`guess_timeout_s`: past it, the plain line — not a second, longer wait."""
    reasoner = ThinkingReasoner(json.dumps({"answer": "Paris, sir."}), guess=TimeoutError("hung"))
    orch = make_orch(reasoner)
    handle(orch, "what is the capital of france")
    assert reasoner.kinds() == ["guess"]
    assert orch._persona.spoken == ["<unknown>"]


def test_a_plan_with_a_step_the_nlu_does_not_know_is_dropped_whole():
    thought = json.dumps({"commands": ["set a timer for 5 minutes", "make the watch sing"]})
    orch = make_orch(ThinkingReasoner(thought), answers=["yes"])
    handle(orch, "wake me in five and serenade me")
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]      # and no question was asked


def test_a_single_planned_command_is_a_did_you_mean():
    """Not a mishearing — it sounds nothing like it — so M4 would not offer
    it. As a confirmed paraphrase it may."""
    orch = make_orch(ThinkingReasoner(json.dumps({"commands": ["find my watch"]})), answers=["yes"])
    handle(orch, "where on earth did i leave my wrist computer")
    assert confirms(orch) == ["<did_you_mean find my watch>"]
    assert labels(orch) == ["find_watch"]


def test_a_planned_chain_may_not_contain_a_meta_action():
    thought = json.dumps({"commands": ["find my watch", "remove the timer skill"]})
    orch = make_orch(ThinkingReasoner(thought), answers=["yes"])
    handle(orch, "ring the watch and get rid of timers")
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]


def test_planning_can_be_switched_off():
    orch = make_orch(
        ThinkingReasoner(PLAN), answers=["yes"],
        config=configured(reasoner={"plan_commands": False}),
    )
    handle(orch, "wake me in five and beep the watch")
    assert orch.registry.calls == []
    assert orch._persona.spoken == ["<unknown>"]


# -- answering -----------------------------------------------------------------------

def test_an_open_question_is_answered_in_the_reasoner_s_own_words():
    orch = make_orch(ThinkingReasoner(json.dumps({"answer": "Paris, sir."})))
    handle(orch, "what is the capital of france")
    assert orch._persona.spoken == ["Paris, sir."]
    assert orch._persona.phrased == []     # already in character: not a third call
    assert orch.registry.calls == []


def test_a_long_answer_is_cut_to_something_speakable():
    long = " ".join(f"Sentence number {n} goes on for a while." for n in range(30))
    orch = make_orch(ThinkingReasoner(json.dumps({"answer": long})))
    handle(orch, "tell me everything")
    (said,) = orch._persona.spoken
    assert said.startswith("Sentence number 0")
    assert len(said) <= reasoning.MAX_ANSWER_CHARS
    assert said.endswith(".")              # cut at a sentence, not mid-word


@pytest.mark.parametrize(
    "thought",
    ['{"none": true}', "", "I am not sure what you mean.", '{"answer": ""}', '{"commands": []}',
     '{"answer": 42}', "[1, 2]", TimeoutError("ollama timed out")],
)
def test_nothing_useful_from_the_reasoner_is_the_plain_line(thought):
    orch = make_orch(ThinkingReasoner(thought), answers=["yes"])
    handle(orch, "flibber")
    assert orch._persona.spoken == ["<unknown>"]
    assert orch.registry.calls == []


def test_answering_can_be_switched_off():
    orch = make_orch(
        ThinkingReasoner(json.dumps({"answer": "Paris, sir."})),
        config=configured(reasoner={"answer_questions": False}),
    )
    handle(orch, "what is the capital of france")
    assert orch._persona.spoken == ["<unknown>"]


def test_with_both_switched_off_the_reasoner_is_not_asked_to_think():
    reasoner = ThinkingReasoner(json.dumps({"answer": "Paris, sir."}))
    orch = make_orch(
        reasoner, config=configured(reasoner={"answer_questions": False, "plan_commands": False})
    )
    handle(orch, "what is the capital of france")
    assert reasoner.kinds() == ["guess"]
    assert orch._persona.spoken == ["<unknown>"]


def test_without_a_reasoner_nothing_changes():
    orch = make_orch(reasoner=None)
    handle(orch, "what is the capital of france")
    assert orch._persona.spoken == ["<unknown>"]


def test_an_unavailable_reasoner_is_never_asked():
    reasoner = ThinkingReasoner(json.dumps({"answer": "Paris, sir."}), available=False)
    orch = make_orch(reasoner)
    handle(orch, "what is the capital of france")
    assert reasoner.calls == []
    assert orch._persona.spoken == ["<unknown>"]


def test_a_low_confidence_meta_action_can_be_answered_too():
    orch = make_orch(ThinkingReasoner(json.dumps({"answer": "Paris, sir."})))
    handle(orch, "what is the capital of france", label="revert_skill", confidence=0.4)
    assert orch._persona.spoken == ["Paris, sir."]


# -- M4 keeps its place ---------------------------------------------------------------

def test_a_declined_mishearing_does_not_fall_through_to_an_answer():
    reasoner = ThinkingReasoner(json.dumps({"answer": "Corn is a cereal, sir."}), guess="flip a coin")
    orch = make_orch(reasoner, answers=["no"])
    handle(orch, "flip a corn")
    assert orch._persona.spoken == ["<did_you_mean flip a coin>", "<unknown>"]
    assert reasoner.kinds() == ["guess"]


def test_the_mishearing_check_can_be_off_with_answering_on():
    reasoner = ThinkingReasoner(json.dumps({"answer": "Paris, sir."}), guess="flip a coin")
    orch = make_orch(reasoner, config=configured(reasoner={"correct_misheard": False}))
    handle(orch, "what is the capital of france")
    assert reasoner.kinds() == ["thought"]
    assert orch._persona.spoken == ["Paris, sir."]


# -- what the reasoner is told ---------------------------------------------------------

def test_the_prompt_carries_the_commands_the_memory_and_the_time(tmp_path):
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    memory.device("local").add_turn("what is the capital of france", "Paris, sir.")
    memory.device("kitchen").remember("the oven is broken")
    memory.device("kitchen").add_turn("is the oven on", "No, sir.")

    reasoner = ThinkingReasoner()
    orch = make_orch(reasoner, memory=memory)
    handle(orch, "and where did i park")

    kind, system, prompt, options = reasoner.calls[-1]
    assert kind == "thought"
    assert "and where did i park" in prompt
    assert "find_watch" in prompt and "find my watch" in prompt      # what it can do
    assert "I parked on level two" in prompt                          # what it was told
    assert "what is the capital of france" in prompt and "Paris, sir." in prompt
    assert str(dt.date.today().year) in prompt                        # when it is
    # another device's memory is not this device's business
    assert "oven" not in prompt
    # in character, and machine-readable
    assert "J.A.R.V.I.S." in system and "JSON" in system
    assert options.get("format") == "json"
    assert options.get("timeout") == orch.config.reasoner.reason_timeout_s


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"answer": "Paris, sir."}', reasoning.Thought(answer="Paris, sir.")),
        ('{"commands": ["find my watch"]}', reasoning.Thought(commands=("find my watch",))),
        ('{"commands": ["Set a timer for 5 minutes.", " find my watch "]}',
         reasoning.Thought(commands=("Set a timer for 5 minutes", "find my watch"))),
        # a model that wraps its JSON in prose or a code fence
        ('Sure!\n```json\n{"answer": "Paris."}\n```', reasoning.Thought(answer="Paris.")),
        # commands win over a narrated answer
        ('{"commands": ["find my watch"], "answer": "Ringing it."}',
         reasoning.Thought(commands=("find my watch",))),
        # a model that fills in every key it was shown
        ('{"commands": [], "answer": "Paris, sir."}', reasoning.Thought(answer="Paris, sir.")),
        ('{"commands": null, "answer": "Paris, sir.", "none": false}', reasoning.Thought(answer="Paris, sir.")),
        ('{"none": true}', None),
        ('{"answer": "   "}', None),
        ('{"commands": "find my watch"}', None),          # not a list
        ('{"commands": ["a", "b", "c", "d", "e", "f"]}', None),   # more than a command chain holds
        ('{"commands": ["find my watch", 7]}', None),
        ("not json at all", None),
        ("", None),
    ],
)
def test_parse(reply, expected):
    assert reasoning.parse(reply) == expected


def test_build_prompt_leaves_out_what_there_is_none_of():
    prompt = reasoning.build_prompt("hello", ["search: search <something>"], [], [], dt.datetime(2026, 10, 1, 21, 5))
    assert "remember" not in prompt.lower()
    assert "Earlier" not in prompt
    assert "Thursday" in prompt and "21:05" in prompt


# -- reading back and forgetting ----------------------------------------------------------

def test_what_do_you_remember_reads_the_facts_back(tmp_path):
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    memory.device("local").remember("the wifi code is 1234")
    orch = make_orch(memory=memory)
    handle(orch, "what do you remember", label="recall_memory", confidence=0.9)
    (said,) = orch._persona.spoken
    assert "I parked on level two" in said and "the wifi code is 1234" in said


def test_nothing_to_read_back():
    orch = make_orch()
    handle(orch, "what do you remember", label="recall_memory", confidence=0.9)
    assert orch._persona.spoken == ["<nothing_remembered>"]


def test_forgetting_asks_first_and_then_forgets(tmp_path):
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    memory.device("local").add_turn("hello", "hello, sir")
    orch = make_orch(memory=memory, answers=["yes"])
    handle(orch, "forget everything i told you", label="forget_memory", confidence=0.9)
    assert orch._persona.spoken == ["<forget_confirm>", "<forgotten>"]
    assert memory.device("local").facts() == [] and memory.device("local").turns() == []


def test_a_no_keeps_everything(tmp_path):
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    orch = make_orch(memory=memory, answers=["no"])
    handle(orch, "forget everything i told you", label="forget_memory", confidence=0.9)
    assert memory.device("local").facts() == ["I parked on level two"]
    assert "<forgotten>" not in orch._persona.spoken


def test_forgetting_needs_the_meta_action_confidence(tmp_path):
    """"forget …" is also how a skill is removed; an unsure one does neither."""
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    orch = make_orch(memory=memory, answers=["yes"])
    handle(orch, "forget the thing", label="forget_memory", confidence=0.45)
    assert orch._persona.spoken == ["<unknown>"]
    assert memory.device("local").facts() == ["I parked on level two"]


def test_a_forget_that_could_not_be_done_is_not_called_forgotten(tmp_path, monkeypatch):
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level two")
    orch = make_orch(memory=memory, answers=["yes"])

    def refuse(self, *a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.unlink", refuse)
    handle(orch, "forget everything i told you", label="forget_memory", confidence=0.9)
    assert orch._persona.spoken == ["<forget_confirm>", "<error>"]


def test_the_new_intents_are_in_the_seed_and_are_inline_actions():
    meta = intent_meta()
    assert meta["recall_memory"].action == "recall_memory"
    assert meta["forget_memory"].action == "forget_memory"
    assert meta["recall_memory"].patterns and meta["forget_memory"].patterns


def test_every_persona_has_the_memory_lines(config):
    from jarvis.core.persona import PERSONAS_DIR, Persona as RealPersona

    for root in sorted(p for p in PERSONAS_DIR.iterdir() if p.is_dir()):
        persona = RealPersona.load(root.name, config)
        for event in ("nothing_remembered", "forget_confirm", "forgotten", "forget_kept"):
            assert persona.line(event), f"persona {root.name!r} has no {event!r} line"
        assert persona.character()


# -- the loop: whose turn, what is recorded ------------------------------------------------

def test_a_turn_is_recorded_under_the_device_it_came_from():
    orch = make_orch()
    orch.standby = True
    orch.mic = FakeMic(script=[True, False])
    orch.mic.device_id = "watch"
    orch.stt = ScriptedSTT(["search black holes"])
    asyncio.run(orch.run())

    (turn,) = orch.memory.device("watch").turns()
    assert turn[0] == "search black holes"
    assert "search" in turn[1]                        # what was said back
    assert orch.memory.device("local").turns() == []


def test_without_a_device_id_the_device_is_local():
    orch = make_orch()
    orch.standby = True
    orch.mic = FakeMic(script=[True, False])
    orch.stt = ScriptedSTT(["search black holes"])
    asyncio.run(orch.run())
    assert len(orch.memory.device("local").turns()) == 1


def test_a_follow_up_can_see_the_turn_before_it(capsys):
    reasoner = ThinkingReasoner(json.dumps({"answer": "Madrid, sir."}))
    orch = make_orch(reasoner)
    orch.standby = True
    orch.mic = FakeMic(script=[True, True, False])
    orch.stt = ScriptedSTT(["search black holes", "and spain"])
    asyncio.run(orch.run())

    prompt = reasoner.calls[-1][2]
    assert "search black holes" in prompt             # the turn before
    assert orch._persona.spoken[-2] == "Madrid, sir."
    assert "answer  :" in capsys.readouterr().out     # why it said that is visible


def test_the_remote_link_says_which_device_is_talking(tmp_path):
    from jarvis.remote.server import RemoteLink
    from tests.remote_harness import FakeVoder, make_config

    link = RemoteLink(make_config(tmp_path), voder=FakeVoder())
    assert link.device_id is None
    connection = SimpleNamespace(device_id="watch")
    link.connect(connection)
    assert link.device_id == "watch"
    link.disconnect(connection)
    assert link.device_id == "watch"     # the turn in flight is still the watch's


def test_one_edge_s_queued_words_are_not_handed_to_the_next_edge(tmp_path):
    """Said to the watch, not yet taken, and the watch goes away: when the
    kitchen connects it must not run — and remember — the watch's words."""
    from jarvis.remote.server import RemoteLink
    from tests.remote_harness import FakeVoder, make_config

    link = RemoteLink(make_config(tmp_path), voder=FakeVoder())
    watch = SimpleNamespace(device_id="watch")
    link.connect(watch)
    link.submit("remember that the code is 1234")
    link.disconnect(watch)

    link.connect(SimpleNamespace(device_id="watch"))      # the same device, back again
    assert link.triggered(None) is True                    # still its own words

    link.disconnect(link.connection)
    link.connect(SimpleNamespace(device_id="kitchen"))
    assert link.triggered(None) is False
    assert link.device_id == "kitchen"


def test_a_turn_from_an_edge_is_remembered_under_that_edge(tmp_path):
    """The whole remote path: a real socket, the brain's own `RemoteLink` as
    the orchestrator's mic. The memory is the edge's, not `local`'s."""
    from jarvis.remote import protocol as P
    from tests.remote_harness import DEVICE, Brain, FakeTranscriber, make_config
    from tests.test_remote_link import connected

    async def scenario():
        brain = Brain(make_config(tmp_path), transcriber=FakeTranscriber("Jarvis, search black holes"))
        await brain.start()
        try:
            async with connected(brain) as edge:
                await edge.say()
                speech = await edge.expect("speech")
                await edge.send(P.control(P.CONTROL.PLAYBACK_DONE, id=speech["id"]))
                await edge.expect("state")
                memory = brain.orchestrator.memory
                for _ in range(100):  # the turn is recorded once the answer has been spoken
                    if memory.device(DEVICE).turns():
                        break
                    await asyncio.sleep(0.02)
                (turn,) = memory.device(DEVICE).turns()
                assert turn[0] == "search black holes"
                assert turn[1]
                assert memory.device("local").turns() == []
        finally:
            await brain.stop()

    asyncio.run(asyncio.wait_for(scenario(), timeout=20))


def test_a_cancel_during_a_question_inside_a_turn_does_not_stop_the_loop():
    """Known issue #5: "set a timer" -> "How long, sir?" -> Enter used to
    take `run()` down."""
    orch = make_orch()
    orch.nlu.mapping["set a timer"] = ("set_timer", 0.9)
    orch.standby = True
    orch.stt = ScriptedSTT(["set a timer", "five minutes"])

    class CancelledAtTheQuestion(FakeMic):
        def record_utterance(self, *a, **k):
            if self._i == 1:  # the capture that would hear the answer
                orch._interrupter.cancel_flag.set()
            return super().record_utterance(*a, **k)

    orch.mic = CancelledAtTheQuestion(script=[True, False, False])
    asyncio.run(orch.run())                           # returns: the loop survived

    assert orch.registry.calls == []
    assert "<standby>" in orch._persona.spoken        # and the session ended the normal way


# -- the Ollama client -------------------------------------------------------------------

def test_generate_can_ask_for_json(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"none": true}'}

    def fake_post(url, json, timeout):
        sent.update(json=json)
        return Response()

    monkeypatch.setattr("jarvis.core.reasoner.requests.post", fake_post)
    r = Reasoner(base_url="http://ollama.test", model="m")
    r.generate("sys", "prompt", format="json")
    assert sent["json"]["format"] == "json"
    r.generate("sys", "prompt")
    assert "format" not in sent["json"]
