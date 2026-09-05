"""`RetrainWorker` — the retrain runs in a real subprocess. Needs the fastembed
model; skips like the other NLU tests when it can't be fetched (see
`tests/conftest.py`'s `embedder` fixture)."""

from __future__ import annotations

from jarvis.nlu.corpus import Example
from jarvis.nlu.retrain_worker import RetrainWorker

EXAMPLES = [
    Example(text="search black holes", label="search", source="seed"),
    Example(text="search cats", label="search", source="seed"),
    Example(text="open github", label="open_app", source="seed"),
    Example(text="open youtube", label="open_app", source="seed"),
]


def test_retrain_worker_trains_in_a_subprocess(embedder, embedding_model, tmp_path):
    worker = RetrainWorker()
    worker.start(EXAMPLES, embedding_model, tmp_path)
    result = worker.poll(60.0)
    assert result is not None, "retrain worker timed out"
    kind, payload = result
    assert kind == "ok", payload
    assert payload.n_examples == len(EXAMPLES)
    assert (payload.path / "head.joblib").is_file()


def test_poll_returns_none_before_the_worker_finishes(embedder, embedding_model, tmp_path):
    worker = RetrainWorker()
    worker.start(EXAMPLES, embedding_model, tmp_path)
    assert worker.poll(0.0) is None
    # drain it so the test doesn't leak a running process
    worker.poll(60.0)
