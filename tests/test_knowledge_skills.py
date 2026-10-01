"""M5: the skills over the knowledge base — `recall`, `remember`, and the two
that changed (`note` mirrors into the docs dir, `search` asks the notes first).

The answer has two tiers and neither is Claude: Ollama composes it when it is
up, otherwise the best snippet is read out in a canned frame with its source.
"""

from __future__ import annotations

import datetime as dt

import pytest

from jarvis.core.context import Context
from jarvis.nlu import slots
from jarvis.skills.builtin import note, recall, remember, search
from jarvis.skills.registry import BUILTIN_PACKAGE, Registry
from tests.knowledge_harness import FakeReasoner, make_knowledge

DESCALE = "Descale the coffee machine every two months with citric acid."
QUESTION = "how often do i descale the coffee machine"


def _ctx(config, tmp_path, knowledge=None, llm=None):
    return Context(
        say=lambda _t: None, config=config, _data_dir=tmp_path / "skill",
        llm=llm, knowledge=knowledge,
    )


@pytest.fixture
def kb(tmp_path):
    knowledge = make_knowledge(tmp_path)
    knowledge.store.replace_source("/home/sir/docs/coffee-manual.md", "1:1", [
        DESCALE,
        "The water tank holds one and a half litres.",
    ])
    return knowledge


# -- recall: the verbatim-snippet tier ----------------------------------------

def test_recall_reads_the_best_snippet_and_names_its_source(config, tmp_path, kb):
    line = recall.run(_ctx(config, tmp_path, kb), query=QUESTION)
    assert line == (
        "From your notes, sir: Descale the coffee machine every two months "
        "with citric acid — from coffee manual."
    )


def test_a_spoken_fact_is_sourced_by_its_date(config, tmp_path, kb):
    kb.remember("i parked on level three", dt.datetime(2026, 10, 1, 9, 30))
    line = recall.run(_ctx(config, tmp_path, kb), query="where did i park")
    assert line == (
        "From your notes, sir: i parked on level three — from what you told me on 1 October."
    )


def test_of_two_equally_good_facts_the_newer_one_is_the_answer(config, tmp_path, kb):
    """Yesterday's parking level is not where the car is today."""
    kb.remember("i parked on level three", dt.datetime(2026, 9, 30, 8, 0))
    kb.remember("i parked on level five", dt.datetime(2026, 10, 1, 8, 0))
    line = recall.run(_ctx(config, tmp_path, kb), query="where did i park")
    assert "level five" in line and "level three" not in line
    assert "1 October" in line


def test_a_long_chunk_is_cut_to_whole_sentences(config, tmp_path):
    knowledge = make_knowledge(tmp_path)
    sentences = [f"Maintenance step {i} is to check valve {i} carefully." for i in range(30)]
    knowledge.store.replace_source("/docs/service.md", "1:1", [" ".join(sentences)])

    line = recall.run(_ctx(config, tmp_path, knowledge), query="maintenance step check valve")

    spoken = line.removeprefix("From your notes, sir: ").removesuffix(" — from service.")
    assert spoken != line
    assert len(spoken) <= 320
    assert spoken.startswith(sentences[0].rstrip("."))
    # it stops at the end of a sentence, not in the middle of one
    assert (spoken + ".") in " ".join(sentences) + "."
    assert spoken.endswith("carefully")


def test_markdown_heading_marks_are_not_read_out(config, tmp_path):
    knowledge = make_knowledge(tmp_path)
    knowledge.store.replace_source(
        "/docs/house.md", "1:1", ["# Coffee machine Descale it every two months."]
    )
    line = recall.run(_ctx(config, tmp_path, knowledge), query="descale the coffee machine")
    assert "#" not in line
    assert "Coffee machine Descale it every two months" in line


def test_recall_with_nothing_on_the_subject_says_so(config, tmp_path, kb):
    line = recall.run(_ctx(config, tmp_path, kb), query="photosynthesis")
    assert line == "I have nothing on that in your notes, sir."


def test_recall_without_a_question_asks_for_one(config, tmp_path, kb):
    assert "notes" in recall.run(_ctx(config, tmp_path, kb), query="").lower()


def test_recall_with_the_knowledge_base_off_says_so(config, tmp_path):
    line = recall.run(_ctx(config, tmp_path, knowledge=None), query=QUESTION)
    assert "switched off" in line


# -- recall: the composed tier (Ollama) ---------------------------------------

def test_recall_composes_from_the_chunks_when_a_reasoner_is_up(config, tmp_path, kb):
    reply = "Every two months, sir, with citric acid — according to the coffee manual."
    llm = FakeReasoner(reply)

    line = recall.run(_ctx(config, tmp_path, kb, llm), query=QUESTION)

    assert line == reply
    [(system, prompt)] = llm.calls
    assert "only" in system.lower()          # told to stay inside the excerpts
    assert DESCALE in prompt                 # ...and given them
    assert "coffee manual" in prompt         # with where each came from
    assert QUESTION in prompt


def test_a_composed_answer_that_forgot_its_source_is_given_one(config, tmp_path, kb):
    llm = FakeReasoner("Every two months, with citric acid.")
    line = recall.run(_ctx(config, tmp_path, kb, llm), query=QUESTION)
    assert line == "Every two months, with citric acid — from coffee manual."


@pytest.mark.parametrize(
    "llm",
    [
        FakeReasoner("", available=True),                        # said nothing
        FakeReasoner("x", raises=RuntimeError("ollama fell over")),
    ],
    ids=["empty", "raises"],
)
def test_a_reasoner_that_fails_degrades_to_the_snippet(config, tmp_path, kb, llm):
    line = recall.run(_ctx(config, tmp_path, kb, llm), query=QUESTION)
    assert line.startswith("From your notes, sir: Descale the coffee machine")
    assert len(llm.calls) == 1


def test_a_reasoner_that_is_down_is_not_asked(config, tmp_path, kb):
    llm = FakeReasoner("never said", available=False)
    line = recall.run(_ctx(config, tmp_path, kb, llm), query=QUESTION)
    assert line.startswith("From your notes, sir:")
    assert llm.calls == []


def test_the_reasoner_is_not_asked_about_nothing(config, tmp_path, kb):
    """With no chunk to ground it, a model would just make something up."""
    llm = FakeReasoner("Photosynthesis is how plants eat light.")
    line = recall.run(_ctx(config, tmp_path, kb, llm), query="photosynthesis")
    assert line == "I have nothing on that in your notes, sir."
    assert llm.calls == []


def test_a_reasoner_that_finds_no_answer_in_the_chunks_means_no_answer(config, tmp_path, kb):
    """A chunk can clear the bar without answering the question. The model
    says so with a fixed token, and that is "nothing on that" — not a spoken
    "I have nothing on that — from coffee manual"."""
    llm = FakeReasoner("NO_ANSWER")
    line = recall.run(_ctx(config, tmp_path, kb, llm), query=QUESTION)
    assert line == "I have nothing on that in your notes, sir."
    assert len(llm.calls) == 1
    assert "NO_ANSWER" in llm.calls[0][0]  # it was told the token


def test_an_old_line_in_a_file_indexed_today_does_not_beat_a_newer_fact(config, tmp_path):
    knowledge = make_knowledge(tmp_path)
    knowledge.store.replace_source(
        "/docs/dictated-notes.md", "1:1", ["i parked on level two"],
        when=dt.datetime(2026, 9, 1, 8, 0),
    )
    knowledge.remember("i parked on level three", dt.datetime(2026, 9, 30, 8, 0))
    # a new note re-indexes the whole file today
    knowledge.store.replace_source(
        "/docs/dictated-notes.md", "2:2", ["i parked on level two", "buy oat milk"],
        when=dt.datetime(2026, 10, 1, 8, 0),
    )

    line = recall.run(_ctx(config, tmp_path, knowledge), query="where did i park")

    assert "level three" in line


# -- remember ---------------------------------------------------------------------

def test_remember_then_recall(config, tmp_path, kb):
    ctx = _ctx(config, tmp_path, kb)

    assert remember.run(ctx, text="my locker number is 52") == "I'll remember that."

    line = recall.run(ctx, query="my locker number")
    assert "my locker number is 52" in line
    assert kb.store.stats()["facts"] == 1


def test_remember_needs_something_to_remember(config, tmp_path, kb):
    assert "remember" in remember.run(_ctx(config, tmp_path, kb), text="  ").lower()
    assert kb.store.stats()["facts"] == 0


def test_remember_with_the_knowledge_base_off_says_so(config, tmp_path):
    line = remember.run(_ctx(config, tmp_path, knowledge=None), text="the door code is 4821")
    assert "switched off" in line


# -- note mirrors into the docs dir -----------------------------------------------

def test_a_dictated_note_is_recallable_on_the_next_turn(config, tmp_path, kb):
    ctx = _ctx(config, tmp_path, kb)

    assert note.run(ctx, text="the door code is 4821") == "Noted."

    # the notes file is still written, as before...
    assert "the door code is 4821" in (tmp_path / "skill" / "notes.txt").read_text("utf-8")
    # ...and mirrored into the docs dir, where the next scan would find it too
    mirror = kb.docs_dir / "dictated-notes.md"
    assert "the door code is 4821" in mirror.read_text("utf-8")
    # ...and indexed already, without waiting for that scan
    line = recall.run(ctx, query="the door code")
    assert "the door code is 4821" in line
    assert "dictated notes" in line


def test_each_note_comes_back_on_its_own(config, tmp_path, kb):
    ctx = _ctx(config, tmp_path, kb)
    note.run(ctx, text="the door code is 4821")
    note.run(ctx, text="the bins go out on thursday")

    line = recall.run(ctx, query="when do the bins go out")

    assert "bins go out on thursday" in line
    assert "door code" not in line
    # the scan agrees with what `note` indexed: the mirror is not indexed again
    # (the fixture's manual is not a real file, so the scan does drop that)
    result = kb.scan()
    assert result.added == result.updated == result.failed == []


def test_note_still_works_with_the_knowledge_base_off(config, tmp_path):
    ctx = _ctx(config, tmp_path, knowledge=None)
    assert note.run(ctx, text="buy milk") == "Noted."
    assert "buy milk" in (tmp_path / "skill" / "notes.txt").read_text("utf-8")


# -- search: the notes first, then Wikipedia ----------------------------------------

def _no_wikipedia(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("went to Wikipedia")

    monkeypatch.setattr(search, "_search_title", refuse)


def _wikipedia_says(monkeypatch, extract: str):
    monkeypatch.setattr(search, "_search_title", lambda s, q: "Python")
    monkeypatch.setattr(search, "_summary", lambda s, t: {"type": "standard", "extract": extract})


def test_what_is_answers_from_the_notes_when_they_have_it(config, tmp_path, monkeypatch):
    knowledge = make_knowledge(tmp_path, search_min_score=0.6)
    knowledge.store.replace_source("/docs/house.md", "1:1", ["the door code is 4821"])
    _no_wikipedia(monkeypatch)

    line = search.run(_ctx(config, tmp_path, knowledge), query="the door code")

    assert line == "From your notes, sir: the door code is 4821 — from house."


def test_what_is_goes_to_wikipedia_when_the_notes_only_brush_the_subject(
    config, tmp_path, monkeypatch
):
    """A note that merely mentions the word must not beat the encyclopedia:
    `search` uses the stricter `search_min_score`."""
    knowledge = make_knowledge(tmp_path, min_score=0.2, search_min_score=0.6)
    knowledge.store.replace_source("/docs/house.md", "1:1", ["the door code is 4821"])
    assert knowledge.search("python code")  # good enough for `recall`...
    _wikipedia_says(monkeypatch, "Python is a programming language.")

    line = search.run(_ctx(config, tmp_path, knowledge), query="python code")

    assert line == "According to Wikipedia: Python is a programming language."


def test_what_is_goes_to_wikipedia_when_the_reasoner_finds_no_answer_in_the_notes(
    config, tmp_path, monkeypatch
):
    knowledge = make_knowledge(tmp_path, search_min_score=0.6)
    knowledge.store.replace_source("/docs/house.md", "1:1", ["the door code is 4821"])
    _wikipedia_says(monkeypatch, "A door code is a sequence of digits.")

    line = search.run(
        _ctx(config, tmp_path, knowledge, FakeReasoner("NO_ANSWER")), query="the door code"
    )

    assert line == "According to Wikipedia: A door code is a sequence of digits."


def test_search_survives_a_broken_knowledge_base(config, tmp_path, monkeypatch):
    knowledge = make_knowledge(tmp_path)

    def broken(*_a, **_k):
        raise RuntimeError("embedding model missing")

    monkeypatch.setattr(knowledge, "answer", broken)
    _wikipedia_says(monkeypatch, "Python is a programming language.")

    line = search.run(_ctx(config, tmp_path, knowledge), query="python")

    assert line.startswith("According to Wikipedia")


def test_search_with_the_knowledge_base_off_is_wikipedia_as_before(config, tmp_path, monkeypatch):
    _wikipedia_says(monkeypatch, "Python is a programming language.")
    line = search.run(_ctx(config, tmp_path, knowledge=None), query="python")
    assert line.startswith("According to Wikipedia")


# -- the registry hands it to skills ------------------------------------------------

def test_the_registry_gives_skills_the_knowledge_base(config, kb):
    reg = Registry.discover(config, packages=(BUILTIN_PACKAGE,), knowledge=kb)

    assert {"recall", "remember"} <= set(reg.names())
    assert "Descale the coffee machine" in reg.dispatch("recall", {"query": QUESTION})
    # a rebuilt registry (after a skill is learned) still has it
    assert "Descale the coffee machine" in reg.rebuilt().dispatch("recall", {"query": QUESTION})


def test_the_new_skills_declare_what_they_touch():
    assert recall.MANIFEST.permissions == frozenset({"fs_read"})
    assert remember.MANIFEST.permissions == frozenset({"fs_write"})
    assert recall.MANIFEST.required_params == ["query"]


# -- slots ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,text,expected",
    [
        ("recall", "what do my notes say about the wifi code", {"query": "the wifi code"}),
        ("recall", "jarvis what did i tell you about the meeting", {"query": "the meeting"}),
        ("recall", "check my notes for the door code please", {"query": "the door code"}),
        # no lead-in to strip: the whole question is the query
        ("recall", "where did i park", {"query": "where did i park"}),
        ("recall", "what does the manual say about descaling",
         {"query": "what does the manual say about descaling"}),
        ("remember", "remember that i parked on level three", {"text": "i parked on level three"}),
        ("remember", "hey jarvis remember my locker number is 52", {"text": "my locker number is 52"}),
        ("remember", "don't forget that the meeting moved to friday",
         {"text": "the meeting moved to friday"}),
        # "remember that" is no longer how a note is dictated
        ("note", "note that the wifi code is 1234", {"text": "the wifi code is 1234"}),
    ],
)
def test_slot_extraction(label, text, expected):
    assert slots.extract(label, text) == expected
