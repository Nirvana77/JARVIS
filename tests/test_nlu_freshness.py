"""Known issue #15: changing `intents.json` or a skill's examples retrains.

`ensure_nlu` used to retrain only when a registered skill was missing from the
model's labels, so edited patterns for intents the model already knew changed
nothing until `python -m jarvis nlu rebuild` — silently. The model now records
a digest of the corpus it was trained on, and a start compares it.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from jarvis.nlu import train as train_mod
from jarvis.nlu.corpus import Example, corpus_digest

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CORPUS = [
    Example("search black holes", "search", "seed"),
    Example("play some lofi", "play", "seed"),
    Example("flip a coin", "flip_a_coin", "skill"),
]


def test_the_digest_follows_the_text_the_labels_and_the_model():
    base = corpus_digest(CORPUS, MODEL)
    assert base == corpus_digest(list(reversed(CORPUS)), MODEL)  # order is not content
    edited = [CORPUS[0], Example("play some jazz", "play", "seed"), CORPUS[2]]
    assert corpus_digest(edited, MODEL) != base
    relabelled = [CORPUS[0], CORPUS[1], Example("flip a coin", "coin", "skill")]
    assert corpus_digest(relabelled, MODEL) != base
    assert corpus_digest(CORPUS, "another/model") != base


def test_training_records_the_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_mod, "_embed",
        lambda texts, model: np.eye(len(texts), 8, dtype=np.float32),
    )
    result = train_mod.train(CORPUS, MODEL, tmp_path)
    meta = json.loads((result.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["corpus_digest"] == corpus_digest(CORPUS, MODEL)


def _trained(cfg, registry, digest):
    """A v1 that knows every registered label, with `digest` in its meta
    (``None``: a meta.json from before the digest existed)."""
    vdir = cfg.nlu_model_dir / "v1"
    vdir.mkdir(parents=True)
    (vdir / "head.joblib").write_bytes(b"")
    (vdir / "labels.json").write_text(json.dumps(registry.names()), encoding="utf-8")
    meta = {"version": 1, "embedding_model": cfg.nlu.embedding_model}
    if digest is not None:
        meta["corpus_digest"] = digest
    (vdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _setup(config, tmp_path, monkeypatch):
    from jarvis import app
    from jarvis.skills.registry import BUILTIN_PACKAGE, Registry

    cfg = dataclasses.replace(config, data_dir=tmp_path)
    registry = Registry.discover(cfg, packages=(BUILTIN_PACKAGE,))
    retrained = []
    monkeypatch.setattr(
        app, "rebuild_nlu",
        lambda c, r: retrained.append(1) or type("R", (), {"version": 2})(),
    )
    return app, cfg, registry, retrained


def test_an_unchanged_corpus_does_not_retrain(config, tmp_path, monkeypatch):
    app, cfg, registry, retrained = _setup(config, tmp_path, monkeypatch)
    _trained(cfg, registry, app.current_corpus_digest(cfg, registry))
    assert app.ensure_nlu(cfg, registry) == 1
    assert retrained == []


def test_a_changed_corpus_retrains(config, tmp_path, monkeypatch):
    app, cfg, registry, retrained = _setup(config, tmp_path, monkeypatch)
    _trained(cfg, registry, "a digest of some older intents.json")
    assert app.ensure_nlu(cfg, registry) == 2
    assert retrained == [1]


def test_a_model_from_before_the_digest_retrains_once(config, tmp_path, monkeypatch):
    app, cfg, registry, retrained = _setup(config, tmp_path, monkeypatch)
    _trained(cfg, registry, None)
    assert app.ensure_nlu(cfg, registry) == 2
    assert retrained == [1]
