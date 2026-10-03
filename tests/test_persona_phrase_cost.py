"""Known issue #16, the half M7 makes live: with the cluster's Ollama up,
``Persona.phrase`` rewrites every skill's reply (measured 0.4–0.9 s each).
It must not build an embedding model per call, nor run on the event loop."""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from jarvis.core import persona as persona_mod
from jarvis.core.persona import Persona
from tests.test_orchestrator import FakeMic, FakeNLU, FakeRegistry, FakeWake


class CountingEmbedding:
    made = 0

    def __init__(self, model_name=None):
        CountingEmbedding.made += 1

    def embed(self, texts):
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            v[len(t) % 4] = 1.0
            yield v


class Up:
    available = True

    def generate(self, system, prompt, **kw):
        return "Heads, sir."


def test_phrase_builds_one_embedder_however_often_it_is_called(config, monkeypatch):
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", CountingEmbedding)
    CountingEmbedding.made = 0
    p = Persona.load("jarvis", config, Up())
    for _ in range(5):
        assert p.phrase("Heads.") == "Heads, sir."
    assert CountingEmbedding.made == 1


def test_the_orchestrator_phrases_off_the_event_loop(config):
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.nlu.corpus import intent_meta

    seen = []

    class Persona_:
        def line(self, event, default=""):
            return f"<{event}>"

        def phrase(self, text):
            seen.append(threading.current_thread() is threading.main_thread())
            return text

    class TTS:
        def say(self, text):
            pass

    o = Orchestrator(
        config=config, wake=FakeWake(), mic=FakeMic(), stt=None, tts=TTS(),
        nlu=FakeNLU({}), persona=Persona_(), registry=FakeRegistry(),
        intent_meta=intent_meta(),
    )
    o.standby = False
    asyncio.run(o.handle("search", "search black holes", 0.9))
    asyncio.run(o.handle("thanks", "thanks", 0.9))
    assert seen and not any(seen)
