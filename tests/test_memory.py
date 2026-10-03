"""M4.5 — per-device memory (`PRD/milestone-4.5-reasoning-and-memory.md`).

Two things per device: the recent conversation (RAM, expires) and lasting
facts (`data/memory/<device>.json`). Everything here is on a tmp dir and a
fake clock.
"""

from __future__ import annotations

import json

import pytest

from jarvis.core.context import Context
from jarvis.core.memory import Memory
from jarvis.skills.builtin import note
from jarvis.skills.registry import BUILTIN_PACKAGE, Registry


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# -- lasting facts -------------------------------------------------------------

def test_facts_survive_a_restart(tmp_path):
    Memory(tmp_path).device("watch").remember("I parked on level two")
    again = Memory(tmp_path)
    assert again.device("watch").facts() == ["I parked on level two"]
    assert (tmp_path / "watch.json").is_file()


def test_facts_are_kept_per_device(tmp_path):
    memory = Memory(tmp_path)
    memory.device("watch").remember("I parked on level two")
    memory.device("livingroom").remember("the wifi code is 1234")
    assert memory.device("watch").facts() == ["I parked on level two"]
    assert memory.device("livingroom").facts() == ["the wifi code is 1234"]
    assert memory.device("kitchen").facts() == []


def test_the_same_fact_twice_is_one_fact_and_the_newest(tmp_path):
    watch = Memory(tmp_path).device("watch")
    watch.remember("I parked on level two")
    watch.remember("the wifi code is 1234")
    watch.remember("i parked on level two.")
    assert watch.facts() == ["the wifi code is 1234", "i parked on level two."]


def test_the_oldest_facts_make_room(tmp_path):
    watch = Memory(tmp_path, max_facts=3).device("watch")
    for n in range(5):
        watch.remember(f"fact {n}")
    assert watch.facts() == ["fact 2", "fact 3", "fact 4"]


def test_nothing_is_remembered_from_nothing(tmp_path):
    watch = Memory(tmp_path).device("watch")
    assert watch.remember("   ") is False
    assert watch.facts() == []
    assert not (tmp_path / "watch.json").exists()


def test_a_fact_is_kept_to_a_sane_length(tmp_path):
    watch = Memory(tmp_path).device("watch")
    watch.remember("x" * 5000)
    assert len(watch.facts()[0]) <= 500


@pytest.mark.parametrize("device", ["../x", "a/b", "", ".hidden", "x" * 80, "wat ch", "watch\n"])
def test_a_device_id_cannot_name_a_file_of_its_own_choosing(tmp_path, device):
    with pytest.raises(ValueError):
        Memory(tmp_path).device(device)


def test_a_damaged_file_reads_as_no_facts_and_is_not_fatal(tmp_path):
    (tmp_path / "watch.json").write_text("{not json", encoding="utf-8")
    watch = Memory(tmp_path).device("watch")
    assert watch.facts() == []
    watch.remember("still works")
    assert json.loads((tmp_path / "watch.json").read_text())["facts"][0]["text"] == "still works"
    # ...and what was there is set aside, not written over: a typo from a
    # hand edit must not cost every fact in the file
    (aside,) = [p for p in tmp_path.iterdir() if p.name != "watch.json"]
    assert aside.read_text(encoding="utf-8") == "{not json"


def test_a_forget_that_cannot_delete_the_file_says_so(tmp_path, monkeypatch):
    watch = Memory(tmp_path).device("watch")
    watch.remember("I parked on level two")

    def refuse(self, *a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.unlink", refuse)
    with pytest.raises(OSError):
        watch.forget()
    assert watch.facts() == ["I parked on level two"]     # still remembered, as it is still on disk


def test_without_a_directory_facts_live_in_ram_only():
    memory = Memory(None)
    memory.device("watch").remember("I parked on level two")
    assert memory.device("watch").facts() == ["I parked on level two"]
    assert Memory(None).device("watch").facts() == []


# -- the recent conversation ---------------------------------------------------

def test_turns_are_kept_per_device_newest_last(tmp_path):
    memory = Memory(tmp_path, turns=3)
    watch = memory.device("watch")
    for n in range(5):
        watch.add_turn(f"q{n}", f"a{n}")
    assert watch.turns() == [("q2", "a2"), ("q3", "a3"), ("q4", "a4")]
    assert memory.device("livingroom").turns() == []


def test_a_conversation_is_forgotten_after_a_quiet_spell(tmp_path):
    clock = Clock()
    watch = Memory(tmp_path, idle_forget_s=600, clock=clock).device("watch")
    watch.add_turn("what is the capital of france", "Paris, sir.")
    clock.now += 599
    assert len(watch.turns()) == 1
    clock.now += 2
    assert watch.turns() == []


def test_turns_are_never_written_to_disk(tmp_path):
    Memory(tmp_path).device("watch").add_turn("where did i park", "Level two, sir.")
    assert list(tmp_path.iterdir()) == []


def test_forget_clears_the_facts_and_the_conversation(tmp_path):
    memory = Memory(tmp_path)
    watch = memory.device("watch")
    watch.remember("I parked on level two")
    watch.remember("the wifi code is 1234")
    watch.add_turn("hello", "hello, sir")
    memory.device("livingroom").remember("untouched")

    assert watch.forget() == 2
    assert watch.facts() == [] and watch.turns() == []
    assert Memory(tmp_path).device("watch").facts() == []
    assert memory.device("livingroom").facts() == ["untouched"]


# -- whose turn it is ------------------------------------------------------------

def test_the_default_device_is_the_one_in_focus(tmp_path):
    memory = Memory(tmp_path)
    assert memory.device().device == "local"
    memory.focus("watch")
    memory.device().remember("I parked on level two")
    assert memory.device("watch").facts() == ["I parked on level two"]
    assert memory.device("local").facts() == []


def test_memory_is_built_from_config(config, tmp_path):
    import dataclasses

    cfg = dataclasses.replace(
        config, data_dir=tmp_path, memory=dataclasses.replace(config.memory, turns=2, max_facts=1)
    )
    memory = Memory.from_config(cfg)
    watch = memory.device("watch")
    watch.remember("a")
    watch.remember("b")
    assert watch.facts() == ["b"]
    assert (tmp_path / "memory" / "watch.json").is_file()


# -- how a fact gets in: the note skill ---------------------------------------------

def test_a_note_is_also_filed_under_the_device_it_was_said_to(config, tmp_path):
    memory = Memory(tmp_path / "memory")
    ctx = Context(
        say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None,
        memory=memory.device("watch"),
    )
    assert note.run(ctx, text="I parked on level two") == "Noted, sir."
    assert memory.device("watch").facts() == ["I parked on level two"]
    # the notes file is written as it always was
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8").rstrip().endswith("I parked on level two")


def test_the_registry_hands_skills_the_memory_of_the_turn_s_device(config, tmp_path):
    memory = Memory(tmp_path)
    reg = Registry.discover(config, packages=(BUILTIN_PACKAGE,), memory=memory)
    memory.focus("watch")
    assert reg._context("note").memory.device == "watch"
    memory.focus("livingroom")
    assert reg._context("note").memory.device == "livingroom"
    # and it survives a rebuild (a skill was learned)
    assert reg.rebuilt().memory is memory


def test_a_registry_without_memory_gives_skills_none(config):
    reg = Registry.discover(config, packages=(BUILTIN_PACKAGE,))
    assert reg._context("note").memory is None
