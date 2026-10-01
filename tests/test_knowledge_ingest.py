"""M5: ingestion — loading files, chunking them, and keeping the store in line
with the docs folder (startup scan + the asyncio interval task)."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import threading

import pytest

from jarvis.knowledge import ingest
from jarvis.knowledge.ingest import chunk, load_text, scan
from tests.knowledge_harness import FakeEmbed, make_knowledge, make_store


def run(coro, timeout=20):
    async def guarded():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(guarded())


def write_pdf(path, text: str) -> None:
    """A one-page PDF saying ``text`` — just enough for pypdf to read back."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Contents")] = writer._add_object(stream)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    with open(path, "wb") as fh:
        writer.write(fh)


def scan_docs(store, docs, **kw):
    kw.setdefault("chunk_chars", 800)
    kw.setdefault("chunk_overlap", 100)
    return scan(store, docs, **kw)


@pytest.fixture
def docs(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    return path


# -- chunking -------------------------------------------------------------------

def test_a_short_text_is_one_chunk():
    assert chunk("The wifi code is 1234.") == ["The wifi code is 1234."]


def test_nothing_to_chunk():
    assert chunk("") == []
    assert chunk("  \n\n \t ") == []


def test_paragraphs_are_chunks_of_their_own():
    """One dictated note per paragraph must come back as that note, not as a
    slab of every note around it."""
    text = (
        "The door code is 4821 and it changes every January.\n\n"
        "The bins go out on Thursday night, recycling every other week.\n"
    )
    assert chunk(text) == [
        "The door code is 4821 and it changes every January.",
        "The bins go out on Thursday night, recycling every other week.",
    ]


def test_a_heading_stays_with_the_paragraph_under_it():
    text = "# Coffee machine\n\nDescale it every two months with citric acid.\n"
    assert chunk(text) == ["Coffee machine: Descale it every two months with citric acid."]


def test_headings_stack_and_a_last_one_is_not_lost():
    text = "# House\n\n## Coffee machine\n\nDescale it every two months.\n\n# Garden\n"
    assert chunk(text) == ["House: Coffee machine: Descale it every two months.", "Garden"]


def test_a_heading_needs_no_blank_line_under_it():
    """The usual way to write markdown: each section is its own chunk."""
    text = (
        "# Wifi\nThe code is 1234.\n\n# Boiler\nService due in May.\n\n"
        "# Car\r\nParked on level 3.\r\n\r\nA later paragraph."
    )
    assert chunk(text) == [
        "Wifi: The code is 1234.",
        "Boiler: Service due in May.",
        "Car: Parked on level 3.",
        "A later paragraph.",
    ]


def test_a_hash_without_a_space_is_not_a_heading():
    assert chunk("#todo buy milk\n\nThe bins go out on Thursday.") == [
        "#todo buy milk",
        "The bins go out on Thursday.",
    ]


def test_a_nonsense_chunk_size_still_terminates():
    chunks = chunk("hello world", max_chars=0, overlap=50)
    assert "".join(chunks) == "helloworld"


def test_line_breaks_inside_a_paragraph_are_just_spaces():
    assert chunk("Descale it every\ntwo months\twith  citric acid, please.") == [
        "Descale it every two months with citric acid, please."
    ]


def test_a_long_paragraph_is_split_within_the_limit_and_overlaps():
    sentences = [f"Sentence number {i} is about maintenance step {i}." for i in range(40)]
    chunks = chunk(" ".join(sentences), max_chars=300, overlap=80)

    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    # nothing is lost...
    for sentence in sentences:
        assert any(sentence in c for c in chunks)
    # ...and each chunk opens with the tail of the one before it
    for before, after in zip(chunks, chunks[1:]):
        carried = after.split(". ")[0] + "."
        assert carried in before


def test_a_sentence_longer_than_the_limit_is_still_split():
    chunks = chunk("word " * 400, max_chars=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert sum(c.count("word") for c in chunks) == 400


# -- loaders --------------------------------------------------------------------

def test_text_and_markdown_are_read_directly(docs):
    (docs / "a.txt").write_text("plain text", encoding="utf-8")
    (docs / "b.md").write_text("# markdown\n\nbody", encoding="utf-8")
    assert load_text(docs / "a.txt") == "plain text"
    assert load_text(docs / "b.md") == "# markdown\n\nbody"


def test_a_pdf_is_read_through_pypdf(docs):
    write_pdf(docs / "manual.pdf", "Descale the machine every two months.")
    assert "Descale the machine every two months." in load_text(docs / "manual.pdf")


def test_bytes_that_are_not_utf8_do_not_stop_a_text_file(docs):
    (docs / "latin.txt").write_bytes("caf\xe9 opens at nine".encode("latin-1"))
    assert "opens at nine" in load_text(docs / "latin.txt")


# -- scanning -------------------------------------------------------------------

def test_a_scan_ingests_the_folder(tmp_path, docs):
    (docs / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    (docs / "house").mkdir()
    (docs / "house" / "bins.txt").write_text("The bins go out on Thursday.", encoding="utf-8")
    write_pdf(docs / "manual.pdf", "Descale the machine every two months.")
    store = make_store(tmp_path)

    result = scan_docs(store, docs)

    assert sorted(result.added) == sorted(
        str(docs / p) for p in ("wifi.md", "house/bins.txt", "manual.pdf")
    )
    assert result.updated == result.removed == result.failed == []
    assert result.changed
    [hit] = store.search("when do the bins go out", k=1)
    assert hit.text == "The bins go out on Thursday."
    assert hit.source == str(docs / "house" / "bins.txt")
    assert "Descale" in store.search("descale the machine", k=1)[0].text


def test_it_only_reads_what_it_knows_how_to_read(tmp_path, docs):
    (docs / "notes.md").write_text("The spare key is under the blue pot.", encoding="utf-8")
    (docs / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (docs / "data.json").write_text('{"spare": "key"}', encoding="utf-8")
    (docs / ".hidden.md").write_text("a dotfile", encoding="utf-8")
    (docs / ".git").mkdir()
    (docs / ".git" / "description.txt").write_text("inside a dot directory", encoding="utf-8")
    store = make_store(tmp_path)

    result = scan_docs(store, docs)

    assert result.added == [str(docs / "notes.md")]


def test_a_second_scan_of_an_unchanged_folder_does_nothing(tmp_path, docs):
    (docs / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    embed = FakeEmbed()
    store = make_store(tmp_path, embed)
    scan_docs(store, docs)
    embedded = len(embed.calls)

    result = scan_docs(store, docs)

    assert not result.changed
    assert len(embed.calls) == embedded


def test_a_changed_file_is_indexed_again(tmp_path, docs):
    wifi = docs / "wifi.md"
    wifi.write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    scan_docs(store, docs)

    wifi.write_text("The wifi code is 987654.", encoding="utf-8")
    result = scan_docs(store, docs)

    assert result.updated == [str(wifi)]
    assert result.added == []
    assert [h.text for h in store.search("wifi code", k=5)] == ["The wifi code is 987654."]


def test_a_touched_file_of_the_same_size_is_indexed_again(tmp_path, docs):
    wifi = docs / "wifi.md"
    wifi.write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    scan_docs(store, docs)

    wifi.write_text("The wifi code is 9876.", encoding="utf-8")  # same length
    stat = wifi.stat()
    os.utime(wifi, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert scan_docs(store, docs).updated == [str(wifi)]
    assert store.search("wifi code", k=1)[0].text == "The wifi code is 9876."


def test_a_removed_file_is_dropped_and_spoken_facts_are_not(tmp_path, docs):
    wifi = docs / "wifi.md"
    wifi.write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    store.add_fact("i parked on level three")
    scan_docs(store, docs)

    wifi.unlink()
    result = scan_docs(store, docs)

    assert result.removed == [str(wifi)]
    assert [h.text for h in store.search("wifi code parked", k=5)] == ["i parked on level three"]


def test_a_missing_folder_is_an_empty_folder(tmp_path):
    store = make_store(tmp_path)
    missing = tmp_path / "nowhere"

    result = scan_docs(store, missing)

    assert not result.changed
    assert not missing.exists()                    # and it is not created
    assert not (tmp_path / "kb.sqlite").exists()   # nor is the database


def test_a_file_that_cannot_be_read_does_not_stop_the_others(tmp_path, docs):
    (docs / "broken.pdf").write_bytes(b"this is not a pdf at all")
    (docs / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    embed = FakeEmbed()
    store = make_store(tmp_path, embed)

    result = scan_docs(store, docs)

    assert result.failed == [str(docs / "broken.pdf")]
    assert result.added == [str(docs / "wifi.md")]
    assert store.search("wifi code", k=1)[0].text == "The wifi code is 1234."
    # it is not retried (and not complained about) every interval, only once
    # the file changes
    again = scan_docs(store, docs)
    assert again.failed == [] and not again.changed


def _not_root():
    return os.geteuid() != 0


def test_a_folder_that_went_away_does_not_empty_the_index(tmp_path, docs):
    """A docs folder on a drive that dropped out is not a folder whose files
    were all deleted: the index is kept, and nothing is re-embedded when the
    folder comes back."""
    (docs / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    embed = FakeEmbed()
    store = make_store(tmp_path, embed)
    scan_docs(store, docs)
    embedded = len(embed.calls)

    away = tmp_path / "unmounted"
    docs.rename(away)
    result = scan_docs(store, docs)

    assert result.unavailable and result.removed == []
    assert store.search("wifi code", k=1)[0].text == "The wifi code is 1234."

    away.rename(docs)
    result = scan_docs(store, docs)
    assert not result.unavailable and not result.changed
    assert len(embed.calls) == embedded + 1  # only the search above


@pytest.mark.skipif(not _not_root(), reason="root can read anything")
def test_a_folder_that_cannot_be_listed_does_not_empty_the_index(tmp_path, docs):
    (docs / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    scan_docs(store, docs)

    docs.chmod(0o000)
    try:
        result = scan_docs(store, docs)
    finally:
        docs.chmod(0o755)

    assert result.unavailable and result.removed == []
    assert store.stats()["files"] == 1


def test_an_emptied_folder_does_empty_the_index(tmp_path, docs):
    wifi = docs / "wifi.md"
    wifi.write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    scan_docs(store, docs)

    wifi.unlink()
    result = scan_docs(store, docs)

    assert result.removed == [str(wifi)] and not result.unavailable


@pytest.mark.skipif(not _not_root(), reason="root can read anything")
def test_a_file_that_stops_being_readable_keeps_what_was_indexed_and_is_retried(tmp_path, docs):
    wifi = docs / "wifi.md"
    wifi.write_text("The wifi code is 1234.", encoding="utf-8")
    store = make_store(tmp_path)
    scan_docs(store, docs)

    wifi.write_text("The wifi code is 987654 now.", encoding="utf-8")
    wifi.chmod(0o000)
    try:
        result = scan_docs(store, docs)
        assert result.failed == [str(wifi)]
        # the last good version is still the answer...
        assert store.search("wifi code", k=1)[0].text == "The wifi code is 1234."
        # ...and the failure is reported once, not every interval
        again = scan_docs(store, docs)
        assert again.failed == [] and not again.changed
    finally:
        wifi.chmod(0o644)

    # readable again — same mtime and size, but it is looked at again
    result = scan_docs(store, docs)
    assert result.updated == [str(wifi)]
    assert store.search("wifi code", k=1)[0].text == "The wifi code is 987654 now."


def test_a_file_dictated_into_mid_scan_ends_up_with_its_newest_text(tmp_path, docs):
    """A scan is embedding the notes file when `note` appends to it and indexes
    it: whichever read the file last must be the one whose text is stored."""
    notes = docs / "dictated-notes.md"
    notes.write_text("the old note about the boiler\n\n", encoding="utf-8")
    started, release = threading.Event(), threading.Event()
    fake = FakeEmbed()

    def gated(texts):
        if not started.is_set():
            started.set()
            assert release.wait(10)
        return fake(texts)

    store = make_store(tmp_path, gated)
    scan = threading.Thread(target=scan_docs, args=(store, docs))
    scan.start()
    assert started.wait(5)  # the scan has read the old text and is embedding it

    with notes.open("a", encoding="utf-8") as fh:
        fh.write("the new note about the door code 4821\n\n")
    dictated = threading.Thread(target=ingest.index_file, args=(store, notes))
    dictated.start()
    release.set()
    scan.join()
    dictated.join()

    assert store.search("door code 4821", k=1)[0].text == "the new note about the door code 4821"
    assert store.file_digests()[str(notes)] == ingest.digest(notes)


def test_a_chunk_is_dated_by_the_file_it_was_first_seen_in(tmp_path, docs):
    old = docs / "manual.md"
    old.write_text("Descale the machine every two months.", encoding="utf-8")
    two_years_ago = dt.datetime(2024, 10, 1, 12, 0)
    os.utime(old, (two_years_ago.timestamp(), two_years_ago.timestamp()))
    store = make_store(tmp_path)

    scan_docs(store, docs)

    assert store.search("descale", k=1)[0].when == two_years_ago


def test_a_scan_stops_between_files_when_told_to(tmp_path, docs):
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text(f"note {name} about something", encoding="utf-8")
    store = make_store(tmp_path)
    stop = threading.Event()
    stop.set()

    result = scan_docs(store, docs, stop=stop)

    assert result.added == []
    assert store.stats()["files"] == 0


# -- the facade ------------------------------------------------------------------

def test_search_only_returns_what_clears_the_bar(tmp_path):
    kb = make_knowledge(tmp_path, min_score=0.5, top_k=3)
    kb.store.replace_source("/docs/a.md", "1:1", [
        "the door code is 4821",
        "the bins go out on thursday",
    ])

    assert [h.text for h in kb.search("the door code")] == ["the door code is 4821"]
    assert kb.search("photosynthesis in plants") == []
    # a caller can ask for a stricter bar than the configured one
    assert kb.search("the door code", min_score=0.99) == []


def test_remember_and_scan_go_through_the_facade(tmp_path):
    kb = make_knowledge(tmp_path)
    kb.docs_dir.mkdir()
    (kb.docs_dir / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")

    kb.remember("  i parked on level three ")
    result = kb.scan()

    assert result.added == [str(kb.docs_dir / "wifi.md")]
    assert kb.search("where did i park")[0].text == "i parked on level three"
    assert kb.search("where did i park")[0].kind == "fact"
    assert kb.store.stats()["chunks"] == 2


# -- the interval task ------------------------------------------------------------

def test_watch_rescans_on_the_interval(tmp_path):
    """The startup scan is done before JARVIS says it is ready (`app`), so the
    task waits one interval before its first."""
    kb = make_knowledge(tmp_path, scan_interval_s=0.05)
    kb.docs_dir.mkdir()
    (kb.docs_dir / "wifi.md").write_text("The wifi code is 1234.", encoding="utf-8")
    before = threading.active_count()

    async def scenario():
        task = asyncio.ensure_future(kb.watch())
        await asyncio.sleep(0)
        assert kb.store.stats()["files"] == 0  # not before the interval is up
        for _ in range(200):
            if kb.store.stats()["files"] == 1:
                break
            await asyncio.sleep(0.01)
        assert kb.store.stats()["files"] == 1

        (kb.docs_dir / "bins.md").write_text("The bins go out on Thursday.", encoding="utf-8")
        for _ in range(200):  # picked up by a later tick, with nobody asking
            if kb.store.stats()["files"] == 2:
                break
            await asyncio.sleep(0.01)
        assert kb.store.stats()["files"] == 2

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert threading.active_count() == before


def test_cancelling_watch_mid_scan_waits_for_the_worker_and_stops_it(tmp_path):
    """A thread cannot be killed: the scan is told to stop at the next file,
    and the task does not finish until its worker has."""
    started, release = threading.Event(), threading.Event()
    fake = FakeEmbed()

    def slow_embed(texts):
        started.set()
        assert release.wait(10)
        return fake(texts)

    kb = make_knowledge(tmp_path, slow_embed, scan_interval_s=0.01)
    kb.docs_dir.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (kb.docs_dir / name).write_text(f"note {name} about something", encoding="utf-8")
    before = threading.active_count()

    async def scenario():
        task = asyncio.ensure_future(kb.watch())
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()  # still waiting for the worker in `embed`
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # the file being embedded was finished; the other two were not started
        assert kb.store.stats()["files"] == 1

    run(scenario())
    assert threading.active_count() == before
    # and the next scan is not still told to stop
    assert len(kb.scan().added) == 2


def test_a_scan_that_blows_up_does_not_end_the_watch(tmp_path, monkeypatch):
    kb = make_knowledge(tmp_path, scan_interval_s=0.02)
    calls = []

    def flaky(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk went away")
        return ingest.ScanResult()

    monkeypatch.setattr(ingest, "scan", flaky)

    async def scenario():
        task = asyncio.ensure_future(kb.watch())
        for _ in range(200):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert len(calls) >= 2


def test_a_second_cancel_still_waits_for_the_worker_and_leaves_scanning_usable(tmp_path):
    """Ctrl-C twice: the worker is still not abandoned, and the stop flag is
    not left set for every later scan."""
    started, release = threading.Event(), threading.Event()
    fake = FakeEmbed()

    def slow_embed(texts):
        started.set()
        assert release.wait(10)
        return fake(texts)

    kb = make_knowledge(tmp_path, slow_embed, scan_interval_s=0.01)
    kb.docs_dir.mkdir()
    for name in ("a.md", "b.md"):
        (kb.docs_dir / name).write_text(f"note {name} about something", encoding="utf-8")
    before = threading.active_count()

    async def scenario():
        task = asyncio.ensure_future(kb.watch())
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert threading.active_count() == before
    assert len(kb.scan().added) == 1
