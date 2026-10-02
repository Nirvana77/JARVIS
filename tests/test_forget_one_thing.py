"""Forgetting one thing, and not remembering a "forget".

Found on the watch, 2026-10-02: "forget about the park" and "remove the park
note" were *stored* — classified as `remember` / `note`, whose slot filler then
kept the whole sentence because it had no lead-in to strip — and "forget
everything" tied with `remove_skill` and was refused. There was also no way to
forget a single fact; only everything.

- `note` and `remember` store something only when the utterance has their own
  lead-in ("remember that …", "note that …"). Otherwise the turn is unclear.
- `forget_fact` ("forget about the park", "remove the park note") finds this
  device's facts that match, asks, and on a yes takes them out of the device's
  memory, the knowledge base, and the notes files.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jarvis.core import forgetting
from jarvis.core.context import Context
from jarvis.core.memory import Memory
from jarvis.nlu import slots
from jarvis.nlu.corpus import intent_meta
from jarvis.skills.builtin import note, remember
from tests.knowledge_harness import make_knowledge
from tests.test_reasoning import make_orch


# -- storing needs its own lead-in --------------------------------------------------

@pytest.mark.parametrize(
    "label, text",
    [
        ("remember", "Remember that I parked on level 2."),
        ("remember", "please remember that the gate code is 7731"),
        ("remember", "don't forget that the bins go out on thursday"),
        ("remember", "keep in mind that my sister's birthday is in march"),
        ("remember", "remember this"),
        ("note", "note that the wifi code is 1234"),
        ("note", "add a note that the boiler was serviced"),
        ("note", "take note that the meeting moved"),
        ("note", "note down that the car needs oil"),
        ("note", "make a note of the plumber's number"),
        ("note", "jot down buy milk"),
        ("note", "write down buy milk"),
        ("note", "take a note"),
    ],
)
def test_a_real_lead_in_is_recognised(label, text):
    assert slots.has_lead_in(label, text)


@pytest.mark.parametrize(
    "label, text",
    [
        ("remember", "Forget about the park."),
        ("remember", "Forget everything about the parking."),
        ("note", "Remove the park note."),
        ("note", "delete the wifi note"),
    ],
)
def test_a_sentence_without_one_is_not_a_lead_in(label, text):
    assert not slots.has_lead_in(label, text)


@pytest.mark.parametrize(
    "label, text",
    [("remember", "Forget about the park."), ("note", "Remove the park note.")],
)
def test_nothing_is_stored_without_a_lead_in(tmp_path, label, text):
    memory = Memory(tmp_path)
    orch = make_orch(memory=memory)
    asyncio.run(orch.handle(label, text, 0.8))
    assert orch.registry.calls == []
    assert memory.device("local").facts() == []
    assert orch._persona.spoken == ["<unknown>"]


def test_with_a_lead_in_it_is_stored_as_before(tmp_path):
    orch = make_orch(memory=Memory(tmp_path))
    asyncio.run(orch.handle("remember", "remember that I parked on level two", 0.9))
    assert orch.registry.calls == [("remember", {"text": "I parked on level two"})]


# -- which facts a "forget about …" means ---------------------------------------------

FACTS = [
    "I parked on level 2",
    "the wifi code is 1234",
    "the gate code is 7731",
    "call the dentist tomorrow",
]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("the park", ["I parked on level 2"]),          # park / parked
        ("parking", ["I parked on level 2"]),
        ("the wifi", ["the wifi code is 1234"]),
        ("the dentist", ["call the dentist tomorrow"]),
        ("the code", ["the wifi code is 1234", "the gate code is 7731"]),
        ("the weather", []),
        ("", []),
        ("the", []),
    ],
)
def test_matching_by_shared_words(query, expected):
    assert forgetting.matching(query, FACTS) == expected


def test_matching_also_uses_meaning_when_an_embedder_is_given():
    """No shared word, but close in meaning: the embedder finds it."""
    vectors = {
        "my car": np.array([1.0, 0.0]),
        "I parked on level 2": np.array([0.9, 0.1]),
        "the wifi code is 1234": np.array([0.0, 1.0]),
    }
    embed = lambda text: vectors[text] / np.linalg.norm(vectors[text])
    assert forgetting.matching("my car", list(vectors)[1:], embed=embed) == ["I parked on level 2"]


def test_at_most_a_few_are_offered():
    facts = [f"code {n} is {n}{n}" for n in range(10)]
    assert len(forgetting.matching("code", facts)) == forgetting.MAX_OFFERED


# -- forgetting one thing --------------------------------------------------------------

def _setup(config, tmp_path):
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    for device, text in [("local", "I parked on level 2"), ("local", "the wifi code is 1234"),
                         ("kitchen", "I parked on level 5")]:
        ctx = Context(say=lambda _t: None, config=config, _data_dir=tmp_path / "skill", llm=None,
                      knowledge=kb, memory=memory.device(device))
        remember.run(ctx, text=text)
    return kb, memory


def test_forget_one_fact_after_a_yes(config, tmp_path):
    kb, memory = _setup(config, tmp_path)
    orch = make_orch(memory=memory, answers=["yes"])
    orch.knowledge = kb

    asyncio.run(orch.handle("forget_fact", "forget about the park", 0.9))

    assert orch._persona.spoken == ["<forget_fact_confirm I parked on level 2>", "<forgotten>"]
    assert memory.device("local").facts() == ["the wifi code is 1234"]
    texts = [h.text for h in kb.search("where did I park", min_score=0.0)]
    assert "I parked on level 2" not in texts
    assert "I parked on level 5" in texts                 # the kitchen's, untouched
    assert "the wifi code is 1234" in [h.text for h in kb.search("wifi code", min_score=0.0)]
    assert len(memory.device("local").knowledge_refs()) == 1


def test_a_no_forgets_nothing(config, tmp_path):
    kb, memory = _setup(config, tmp_path)
    orch = make_orch(memory=memory, answers=["no"])
    orch.knowledge = kb
    asyncio.run(orch.handle("forget_fact", "forget about the park", 0.9))
    assert memory.device("local").facts() == ["I parked on level 2", "the wifi code is 1234"]
    assert orch._persona.spoken[-1] == "<forget_kept>"


def test_nothing_that_matches(config, tmp_path):
    kb, memory = _setup(config, tmp_path)
    orch = make_orch(memory=memory, answers=["yes"])
    orch.knowledge = kb
    asyncio.run(orch.handle("forget_fact", "forget about the weather", 0.9))
    assert orch._persona.spoken == ["<nothing_to_forget>"]
    assert len(memory.device("local").facts()) == 2


def test_several_matches_are_offered_in_one_question(tmp_path):
    memory = Memory(tmp_path)
    memory.device("local").remember("the wifi code is 1234")
    memory.device("local").remember("the gate code is 7731")
    orch = make_orch(memory=memory, answers=["yes"])
    asyncio.run(orch.handle("forget_fact", "forget the code", 0.9))
    assert orch._persona.spoken[0] == (
        "<forget_fact_confirm the wifi code is 1234; the gate code is 7731>"
    )
    assert memory.device("local").facts() == []


def test_a_forgotten_note_leaves_the_notes_files_too(config, tmp_path):
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    notes_dir = tmp_path / "skill"
    ctx = Context(say=lambda _t: None, config=config, _data_dir=notes_dir, llm=None,
                  knowledge=kb, memory=memory.device("local"))
    note.run(ctx, text="the wifi code is 1234")
    note.run(ctx, text="buy milk")

    orch = make_orch(memory=memory, answers=["yes"])
    orch.knowledge = kb
    orch._notes_file = notes_dir / note.NOTES_FILE
    asyncio.run(orch.handle("forget_fact", "remove the wifi note", 0.9))

    assert memory.device("local").facts() == ["buy milk"]
    log = (notes_dir / note.NOTES_FILE).read_text(encoding="utf-8")
    assert "wifi" not in log and "buy milk" in log
    mirror = (kb.docs_dir / note.MIRROR_FILE).read_text(encoding="utf-8")
    assert "wifi" not in mirror and "buy milk" in mirror
    assert not [h for h in kb.search("wifi code", min_score=0.0) if "wifi" in h.text]


def test_forgetting_one_thing_needs_its_own_lead_in(tmp_path):
    """"stop", heard as forget_fact (0.39 on the live model): unclear, not a
    search for facts about "stop"."""
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level 2")
    orch = make_orch(memory=memory, answers=["yes"])
    asyncio.run(orch.handle("forget_fact", "stop", 0.39))
    assert memory.device("local").facts() == ["I parked on level 2"]
    assert orch._persona.spoken == ["<unknown>"]


def test_an_unsure_forget_still_asks_and_names_what_would_go(tmp_path):
    """"forget that I parked on level 2" scored 0.50: it is asked about, not
    refused — the question names exactly what would go."""
    memory = Memory(tmp_path)
    memory.device("local").remember("I parked on level 2")
    orch = make_orch(memory=memory, answers=["yes"])
    asyncio.run(orch.handle("forget_fact", "forget that I parked on level 2", 0.5))
    assert orch._persona.spoken[0] == "<forget_fact_confirm I parked on level 2>"
    assert memory.device("local").facts() == []


def test_the_query_is_what_follows_the_lead_in():
    for text, query in [
        ("forget about the park", "the park"),
        ("Forget everything about the parking.", "the parking"),
        ("remove the park note", "the park"),
        ("delete my note about the wifi", "the wifi"),
        ("forget that I parked on level 2", "I parked on level 2"),
        ("forget what I told you about the dentist", "the dentist"),
    ]:
        assert slots.extract("forget_fact", text) == {"query": query}, text


def test_the_new_intent_is_in_the_seed_and_inline():
    meta = intent_meta()
    assert meta["forget_fact"].action == "forget_fact"
    assert "forget everything" in [p.lower() for p in meta["forget_memory"].patterns]


def test_every_persona_has_the_lines(config):
    from jarvis.core.persona import PERSONAS_DIR, Persona

    for root in sorted(p for p in PERSONAS_DIR.iterdir() if p.is_dir()):
        persona = Persona.load(root.name, config)
        assert "{facts}" in persona.line("forget_fact_confirm"), root.name
        assert persona.line("nothing_to_forget"), root.name
