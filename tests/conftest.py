"""Shared fixtures.

The NLU tests need the fastembed MiniLM model. It is cached after the first
download; when it is genuinely unreachable (offline CI) the ``embedder`` fixture
skips rather than fails.
"""

from __future__ import annotations

import pytest

from jarvis.config import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def embedding_model(config) -> str:
    return config.nlu.embedding_model


@pytest.fixture(scope="session")
def embedder(embedding_model):
    """A ready ``fastembed.TextEmbedding``; skips the test if it can't be built."""
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=embedding_model)
        # force the lazy download/load so the skip happens here, not mid-test
        next(iter(model.embed(["warm up"])))
        return model
    except Exception as exc:  # network down, disk full, etc.
        pytest.skip(f"fastembed model unavailable: {exc!r}")


@pytest.fixture(autouse=True)
def no_leaked_threads():
    """CLAUDE.md step 5: a test that starts a thread must leave none behind.
    M2.5's background jobs run on daemon threads (`jobs.run_detached`), so a
    job that isn't finished or cancelled would show up here."""
    import threading
    import time

    before = {t.ident for t in threading.enumerate()}
    yield
    deadline = time.monotonic() + 1.0  # let just-finished threads exit
    while True:
        leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
        if not leaked or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    assert not leaked, f"test left threads running: {[t.name for t in leaked]}"
