"""Shared fakes for the knowledge-base tests (M5).

The store takes its embedder as a callable, so these tests need no model:
``FakeEmbed`` is a deterministic hashed bag-of-words — texts that share words
are close, texts that share none are orthogonal — and it records every batch
it was asked for, which is how "this chunk was not embedded again" is asserted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re

import numpy as np

from jarvis.config import KnowledgeConfig
from jarvis.knowledge import Knowledge
from jarvis.knowledge.store import KnowledgeStore


class FakeEmbed:
    def __init__(self, dim: int = 256, salt: str = "") -> None:
        self.dim = dim
        self.salt = salt
        self.calls: list[list[str]] = []

    @property
    def texts(self) -> list[str]:
        """Every text embedded so far, in order."""
        return [t for batch in self.calls for t in batch]

    def __call__(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        self.calls.append(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha1((self.salt + token).encode()).hexdigest()
                out[row, int(digest[:8], 16) % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm:
                out[row] /= norm
        return out


class FakeReasoner:
    """Stands in for the Ollama ``Reasoner``: records what it was asked."""

    def __init__(self, reply: str = "", *, available: bool = True, raises=None) -> None:
        self.reply = reply
        self.available = available
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self.raises is not None:
            raise self.raises
        return self.reply


def make_store(tmp_path, embed=None, **kwargs) -> KnowledgeStore:
    return KnowledgeStore(
        tmp_path / "kb.sqlite", embed or FakeEmbed(), model_name="fake", **kwargs
    )


def make_knowledge(tmp_path, embed=None, **settings) -> Knowledge:
    """A ``Knowledge`` over a tmp docs dir and a tmp database."""
    docs = tmp_path / "docs"
    settings.setdefault("docs_dir", str(docs))
    settings.setdefault("min_score", 0.2)
    return Knowledge(make_store(tmp_path, embed), KnowledgeConfig(**settings))


def with_knowledge_paths(config, tmp_path, **settings):
    """The real config, pointed at tmp directories — never the user's own
    ``~/jarvis/knowledge`` or the repo's ``data/``."""
    settings.setdefault("docs_dir", str(tmp_path / "docs"))
    return dataclasses.replace(
        config,
        data_dir=tmp_path / "data",
        knowledge=dataclasses.replace(config.knowledge, **settings),
    )
