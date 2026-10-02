"""M4.5 per-device memory and the M5 knowledge base, together.

Both keep what the speaker said. "Remember that ..." is the knowledge base's
`remember` skill (M5); the device it was said to keeps it too (M4.5), so
"what do you remember?" can read it back and "forget everything I told you"
can take it out of *both* — saying "Forgotten" while the knowledge base can
still answer "where did I park?" would be a lie.
"""

from __future__ import annotations

import asyncio

from jarvis.core.context import Context
from jarvis.core.memory import Memory
from jarvis.skills.builtin import note, remember
from tests.knowledge_harness import make_knowledge
from tests.test_reasoning import make_orch


def _ctx(config, tmp_path, *, knowledge, memory):
    return Context(
        say=lambda _t: None, config=config, _data_dir=tmp_path / "skill", llm=None,
        knowledge=knowledge, memory=memory,
    )


def test_a_remembered_fact_is_kept_by_the_device_and_the_knowledge_base(config, tmp_path):
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    ctx = _ctx(config, tmp_path, knowledge=kb, memory=memory.device("watch"))

    assert remember.run(ctx, text="I parked on level two") == "I'll remember that."

    assert kb.store.stats()["facts"] == 1
    assert memory.device("watch").facts() == ["I parked on level two"]
    assert memory.device("local").facts() == []
    # and the device knows which fact in the knowledge base is its own
    (ref,) = memory.device("watch").knowledge_refs()
    assert ref.startswith("fact:")


def test_with_the_knowledge_base_off_the_device_still_remembers(config, tmp_path):
    memory = Memory(tmp_path / "memory")
    ctx = _ctx(config, tmp_path, knowledge=None, memory=memory.device("watch"))
    assert remember.run(ctx, text="I parked on level two") == "I'll remember that."
    assert memory.device("watch").facts() == ["I parked on level two"]
    assert memory.device("watch").knowledge_refs() == []


def test_a_note_is_filed_with_the_device_but_has_no_fact_of_its_own(config, tmp_path):
    """A note goes into the knowledge base as part of the dictated-notes file,
    not as a fact — there is nothing of its own to take out again."""
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    ctx = _ctx(config, tmp_path, knowledge=kb, memory=memory.device("watch"))
    assert note.run(ctx, text="the wifi code is 1234") == "Noted."
    assert memory.device("watch").facts() == ["the wifi code is 1234"]
    assert memory.device("watch").knowledge_refs() == []


def _remembered(config, tmp_path, kb, memory, device, text):
    remember.run(_ctx(config, tmp_path, knowledge=kb, memory=memory.device(device)), text=text)


def test_forgetting_takes_the_device_s_facts_out_of_the_knowledge_base_too(config, tmp_path):
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    _remembered(config, tmp_path, kb, memory, "local", "I parked on level two")
    _remembered(config, tmp_path, kb, memory, "kitchen", "the oven timer is broken")
    assert kb.search("where did I park")

    orch = make_orch(memory=memory, answers=["yes"])
    orch.knowledge = kb
    asyncio.run(orch.handle("forget_memory", "forget everything i told you", 0.9))

    assert orch._persona.spoken == ["<forget_confirm>", "<forgotten>"]
    assert memory.device("local").facts() == []
    assert not [h for h in kb.search("where did I park") if "level two" in h.text]
    # another device's fact is not this device's to forget
    assert memory.device("kitchen").facts() == ["the oven timer is broken"]
    assert kb.store.stats()["facts"] == 1


def test_a_knowledge_base_that_cannot_forget_is_not_called_forgotten(config, tmp_path, monkeypatch):
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    _remembered(config, tmp_path, kb, memory, "local", "I parked on level two")

    def refuse(path):
        raise OSError("database is locked")

    monkeypatch.setattr(kb.store, "remove_source", refuse)
    orch = make_orch(memory=memory, answers=["yes"])
    orch.knowledge = kb
    asyncio.run(orch.handle("forget_memory", "forget everything i told you", 0.9))

    assert orch._persona.spoken == ["<forget_confirm>", "<error>"]
    assert memory.device("local").facts() == ["I parked on level two"]


def test_a_device_whose_facts_are_only_in_the_knowledge_base_can_still_forget(config, tmp_path):
    """Nothing in RAM, nothing but refs on disk: still something to forget."""
    kb = make_knowledge(tmp_path)
    memory = Memory(tmp_path / "memory")
    _remembered(config, tmp_path, kb, memory, "local", "I parked on level two")
    fresh = Memory(tmp_path / "memory")                 # a restart

    orch = make_orch(memory=fresh, answers=["yes"])
    orch.knowledge = kb
    asyncio.run(orch.handle("forget_memory", "forget everything i told you", 0.9))
    assert kb.store.stats()["facts"] == 0
