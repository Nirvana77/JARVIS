"""NLU pipeline: corpus build, training, thresholded classification, slots."""

from __future__ import annotations

import pytest

from jarvis.nlu import slots
from jarvis.nlu.classifier import UNKNOWN, Classifier
from jarvis.nlu.corpus import build_corpus, expand_pattern, read_corpus_db, write_corpus_db
from jarvis.nlu.train import latest_version, train


# -- corpus -------------------------------------------------------------------

def test_expand_pattern_fills_placeholder():
    out = expand_pattern("Search {intent}")
    assert "Search black holes" in out
    assert all("{intent}" not in phrasing for phrasing in out)
    assert expand_pattern("Goodbye") == ["Goodbye"]


def test_build_corpus_from_seed_has_expected_labels():
    corpus = build_corpus()
    labels = {e.label for e in corpus}
    for expected in ("search", "open_app", "play", "note", "greeting", "goodbye", "shutdown"):
        assert expected in labels
    # placeholder expansion means the search class has many examples
    assert sum(1 for e in corpus if e.label == "search") >= 8


def test_corpus_sqlite_roundtrip(tmp_path):
    corpus = build_corpus()
    db = tmp_path / "corpus.sqlite"
    write_corpus_db(corpus, db)
    back = read_corpus_db(db)
    assert {(e.text, e.label) for e in back} == {(e.text, e.label) for e in corpus}


# -- slots ------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,text,expected",
    [
        ("search", "search black holes", {"query": "black holes"}),
        ("search", "hey jarvis look up the speed of light", {"query": "the speed of light"}),
        ("search", "tell me about neptune please", {"query": "neptune"}),
        ("play", "play some lofi", {"query": "some lofi"}),
        ("open_app", "open github for me", {"app": "github"}),
        ("note", "note that the wifi code is 1234", {"text": "the wifi code is 1234"}),
        ("greeting", "are you awake jarvis", {}),
        ("goodbye", "go to sleep", {}),
    ],
)
def test_slot_extraction(label, text, expected):
    assert slots.extract(label, text) == expected


# -- train + classify -----------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory, embedder, embedding_model):
    out = tmp_path_factory.mktemp("nlu_model")
    corpus = build_corpus()
    result = train(corpus, embedding_model, out)
    assert result.version == 1
    assert latest_version(out) == 1
    clf = Classifier.load(out, embedding_model, threshold=0.35)
    return clf, out, result


def test_seed_intents_classify_above_threshold(trained):
    clf, _, _ = trained
    cases = {
        "search black holes": "search",
        "hey jarvis are you there": "greeting",
        "go to sleep": "goodbye",
        "shut yourself down": "shutdown",
        "play some jazz": "play",
        "open spotify": "open_app",
        "remember that i parked on level three": "note",
    }
    for text, expected in cases.items():
        label, conf = clf.predict(text)
        assert label == expected, f"{text!r} -> {label} ({conf:.2f})"
        assert conf >= clf.threshold


def test_gibberish_is_unknown(trained):
    clf, _, _ = trained
    for junk in ("asdf qwer zxcv", "blorp gnnn wibble frotz", ""):
        label, _ = clf.predict(junk)
        assert label == UNKNOWN


def test_versions_pruned_to_three(trained, embedder, embedding_model):
    _, out, _ = trained
    corpus = build_corpus()
    for _ in range(4):
        train(corpus, embedding_model, out)
    versions = sorted(p.name for p in out.glob("v*"))
    assert len(versions) == 3
    assert versions == ["v3", "v4", "v5"]
