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


def test_worker_is_spawned_not_forked():
    """Known issue #1: forking the live brain (dozens of ONNX Runtime /
    fastembed threads) deadlocked the retrain child — `teach` then waited
    forever in silence. A spawned interpreter inherits no held locks."""
    assert RetrainWorker()._ctx.get_start_method() == "spawn"


def test_a_worker_that_never_answers_is_given_up_on(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from jarvis.core import orchestrator as orch_mod

    terminated = []

    class StuckWorker:
        def start(self, *a):
            pass

        def poll(self, timeout=0.0):
            return None

        def exited(self):
            return False

        def terminate(self):
            terminated.append(True)

    monkeypatch.setattr(orch_mod, "RetrainWorker", StuckWorker)
    monkeypatch.setattr(orch_mod, "RETRAIN_TIMEOUT_S", 0.3)
    config = SimpleNamespace(nlu=SimpleNamespace(embedding_model="m"), nlu_model_dir="d")
    kind, payload = asyncio.run(
        orch_mod.Orchestrator._default_train_and_load(SimpleNamespace(config=config), [])
    )
    assert kind == "error"
    assert "timed out" in payload
    assert terminated
