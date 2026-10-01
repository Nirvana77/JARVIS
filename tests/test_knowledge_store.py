"""M5: the knowledge store — chunks, their embeddings, and the search over them.

Embeddings come from an injected fake (``tests/knowledge_harness.py``), so
nothing here loads a model.
"""

from __future__ import annotations

import datetime as dt
import threading

import pytest

from jarvis.knowledge import store as store_mod
from jarvis.knowledge.store import KnowledgeStore
from tests.knowledge_harness import FakeEmbed, make_store

BACKENDS = pytest.mark.parametrize("use_sqlite_vec", [True, False], ids=["sqlite-vec", "numpy"])

MANUAL = [
    "Descale the coffee machine every two months with citric acid.",
    "The water tank holds one and a half litres.",
    "Clean the milk frother after each use.",
]


def _texts(hits) -> list[str]:
    return [h.text for h in hits]


# -- search ---------------------------------------------------------------------

@BACKENDS
def test_search_ranks_the_matching_chunk_first(tmp_path, use_sqlite_vec):
    store = make_store(tmp_path, use_sqlite_vec=use_sqlite_vec)
    store.replace_source("/docs/manual.md", "1:1", MANUAL)

    hits = store.search("how often should i descale the coffee machine", k=2)

    assert len(hits) == 2
    assert hits[0].text == MANUAL[0]
    assert hits[0].source == "/docs/manual.md"
    assert hits[0].kind == "file"
    assert hits[0].score > hits[1].score


@BACKENDS
def test_the_score_is_cosine_similarity(tmp_path, use_sqlite_vec):
    store = make_store(tmp_path, use_sqlite_vec=use_sqlite_vec)
    store.replace_source("/docs/a.txt", "1:1", ["the door code is 4821", "buy oat milk"])

    hits = store.search("the door code is 4821", k=2)

    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[1].score == pytest.approx(0.0, abs=1e-4)


def test_the_backend_in_use_is_reported(tmp_path):
    assert make_store(tmp_path, use_sqlite_vec=True).backend == "sqlite-vec"
    assert make_store(tmp_path, use_sqlite_vec=False).backend == "numpy"


def test_both_backends_rank_alike(tmp_path):
    chunks = MANUAL + ["The warranty lasts two years.", "Use filtered water in the tank."]
    ranked = []
    for name, use in (("vec", True), ("np", False)):
        (tmp_path / name).mkdir()
        store = make_store(tmp_path / name, use_sqlite_vec=use)
        store.replace_source("/docs/manual.md", "1:1", chunks)
        hits = store.search("how much water does the tank hold", k=5)
        ranked.append([(h.text, round(h.score, 4)) for h in hits])
    assert ranked[0] == ranked[1]


def test_an_empty_store_finds_nothing_embeds_nothing_and_writes_nothing(tmp_path):
    embed = FakeEmbed()
    store = make_store(tmp_path, embed)

    assert store.search("anything at all") == []
    assert store.file_digests() == {}
    assert store.stats()["chunks"] == 0
    assert embed.calls == []
    assert not (tmp_path / "kb.sqlite").exists()


def test_asking_for_more_than_there_is(tmp_path):
    store = make_store(tmp_path)
    store.replace_source("/docs/a.txt", "1:1", ["only one chunk"])
    assert _texts(store.search("one chunk", k=10)) == ["only one chunk"]


# -- sources --------------------------------------------------------------------

@BACKENDS
def test_replacing_a_source_drops_its_old_chunks(tmp_path, use_sqlite_vec):
    store = make_store(tmp_path, use_sqlite_vec=use_sqlite_vec)
    store.replace_source("/docs/wifi.md", "1:1", ["the wifi code is 1234"])
    store.replace_source("/docs/wifi.md", "2:2", ["the wifi code is 9876"])

    assert _texts(store.search("wifi code", k=5)) == ["the wifi code is 9876"]
    assert store.file_digests() == {"/docs/wifi.md": "2:2"}


def test_unchanged_chunks_are_not_embedded_again(tmp_path):
    """`note` appends a line to a growing file on every dictation: that must
    cost one embedding, not the whole file."""
    embed = FakeEmbed()
    store = make_store(tmp_path, embed)
    store.replace_source("/docs/notes.md", "1:1", ["buy milk", "call the dentist"])
    assert embed.texts == ["buy milk", "call the dentist"]

    embedded = store.replace_source(
        "/docs/notes.md", "2:2", ["buy milk", "call the dentist", "book the car in"]
    )

    assert embedded == 1
    assert embed.texts == ["buy milk", "call the dentist", "book the car in"]
    assert _texts(store.search("dentist", k=1)) == ["call the dentist"]


@BACKENDS
def test_removing_a_source(tmp_path, use_sqlite_vec):
    store = make_store(tmp_path, use_sqlite_vec=use_sqlite_vec)
    store.replace_source("/docs/a.txt", "1:1", ["the garage door code is 4821"])
    store.replace_source("/docs/b.txt", "1:1", ["the bins go out on thursday"])

    assert store.remove_source("/docs/a.txt") is True
    assert store.remove_source("/docs/a.txt") is False

    assert _texts(store.search("garage door code", k=5)) == ["the bins go out on thursday"]
    assert store.file_digests() == {"/docs/b.txt": "1:1"}


def test_an_empty_file_is_remembered_so_it_is_not_scanned_forever(tmp_path):
    store = make_store(tmp_path)
    store.replace_source("/docs/empty.txt", "1:0", [])
    assert store.file_digests() == {"/docs/empty.txt": "1:0"}
    assert store.stats()["chunks"] == 0


# -- facts ----------------------------------------------------------------------

def test_a_spoken_fact_is_kept_apart_from_the_files(tmp_path):
    store = make_store(tmp_path)
    when = dt.datetime(2026, 10, 1, 9, 30)
    store.add_fact("i parked on level three", when)
    store.replace_source("/docs/a.txt", "1:1", ["the bins go out on thursday"])

    [hit] = store.search("where did i park", k=1)

    assert hit.text == "i parked on level three"
    assert hit.kind == "fact"
    assert hit.when == when
    # a folder scan works from this, so a fact must never be in it
    assert store.file_digests() == {"/docs/a.txt": "1:1"}
    assert store.stats() == {
        "files": 1, "facts": 1, "chunks": 2, "backend": store.backend,
    }


def test_two_facts_are_two_sources(tmp_path):
    store = make_store(tmp_path)
    store.add_fact("the locker number is 52")
    store.add_fact("the locker number is 52")
    assert store.stats()["facts"] == 2


# -- persistence ----------------------------------------------------------------

@BACKENDS
def test_it_survives_a_reopen(tmp_path, use_sqlite_vec):
    store = make_store(tmp_path, use_sqlite_vec=use_sqlite_vec)
    store.replace_source("/docs/manual.md", "1:1", MANUAL)
    store.close()

    embed = FakeEmbed()
    again = make_store(tmp_path, embed, use_sqlite_vec=use_sqlite_vec)

    assert again.search("milk frother", k=1)[0].text == MANUAL[2]
    assert embed.texts == ["milk frother"]  # the stored vectors were reused
    assert again.file_digests() == {"/docs/manual.md": "1:1"}


def test_a_database_written_without_the_index_is_indexed_on_reopen(tmp_path):
    plain = make_store(tmp_path, use_sqlite_vec=False)
    plain.replace_source("/docs/manual.md", "1:1", MANUAL)
    plain.close()

    indexed = make_store(tmp_path, use_sqlite_vec=True)
    assert indexed.backend == "sqlite-vec"
    assert indexed.search("descale", k=1)[0].text == MANUAL[0]
    indexed.close()

    # ...and what the fallback writes afterwards is not lost on the index
    plain = make_store(tmp_path, use_sqlite_vec=False)
    plain.replace_source("/docs/manual.md", "2:2", ["the grinder has nine settings"])
    plain.close()

    indexed = make_store(tmp_path, use_sqlite_vec=True)
    assert _texts(indexed.search("grinder settings", k=5)) == ["the grinder has nine settings"]


def test_it_falls_back_to_numpy_when_the_extension_will_not_load(tmp_path, monkeypatch):
    def refuse(_conn):
        raise RuntimeError("no loadable extensions in this sqlite build")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", refuse)
    store = make_store(tmp_path, use_sqlite_vec=True)
    store.replace_source("/docs/manual.md", "1:1", MANUAL)

    assert store.backend == "numpy"
    assert store.search("descale", k=1)[0].text == MANUAL[0]


def test_a_changed_embedding_model_re_embeds_everything(tmp_path):
    """Vectors from another model are in another space: searching them with
    the new model's query vector would rank nonsense."""
    old = KnowledgeStore(tmp_path / "kb.sqlite", FakeEmbed(dim=64), model_name="old-model")
    old.replace_source("/docs/manual.md", "1:1", MANUAL)
    old.add_fact("i parked on level three")
    old.close()

    embed = FakeEmbed(dim=128, salt="another space")
    new = KnowledgeStore(tmp_path / "kb.sqlite", embed, model_name="new-model")
    hits = new.search("where did i park", k=1)

    assert hits[0].text == "i parked on level three"
    assert sorted(embed.calls[0]) == sorted(MANUAL + ["i parked on level three"])
    # once is enough
    new.search("descale", k=1)
    assert len(embed.calls) == 3


# -- threads --------------------------------------------------------------------

def test_it_is_usable_from_another_thread(tmp_path):
    """Skills run in `asyncio.to_thread` workers, scans in others."""
    store = make_store(tmp_path)
    store.replace_source("/docs/manual.md", "1:1", MANUAL)
    found: list[str] = []
    errors: list[BaseException] = []

    def work():
        try:
            store.add_fact("the spare key is under the blue pot")
            found.extend(_texts(store.search("spare key", k=1)))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    before = threading.active_count()
    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert found == ["the spare key is under the blue pot"] * 4
    assert threading.active_count() == before
