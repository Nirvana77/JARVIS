"""Inference: embed a transcript, run the sklearn head, apply the threshold.

Replaces ``libs/brain.py``. ``predict`` returns ``(label, confidence)``; a max
class probability below ``nlu.threshold`` becomes ``("unknown", confidence)`` so
the orchestrator can fall back (M2: offer to learn the skill).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from jarvis.nlu.train import latest_version

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Prediction:
    label: str                       # the chosen intent, or "unknown"
    confidence: float                # probability of the top class
    ranking: list[tuple[str, float]] # every class, best first
    similarity: float                # cosine to the nearest training phrase

    def __iter__(self):              # so `label, conf = clf.predict(text)` still works
        return iter((self.label, self.confidence))


class Classifier:
    def __init__(
        self,
        embedder,
        head,
        labels: list[str],
        threshold: float,
        version: int,
        embedding_model: str,
        train_embeddings: np.ndarray | None = None,
        similarity_floor: float = 0.30,
    ) -> None:
        self._embedder = embedder
        self._head = head
        self.labels = labels
        self.threshold = threshold
        self.version = version
        self.embedding_model = embedding_model
        self._train_emb = train_embeddings
        self.similarity_floor = similarity_floor

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        embedding_model: str,
        threshold: float,
        similarity_floor: float = 0.30,
    ) -> "Classifier":
        model_dir = Path(model_dir)
        version = latest_version(model_dir)
        if version is None:
            raise FileNotFoundError(
                f"no trained NLU model under {model_dir} — run `python -m jarvis "
                f"nlu rebuild` (or start jarvis once to auto-train v1)"
            )
        vdir = model_dir / f"v{version}"
        head = joblib.load(vdir / "head.joblib")
        labels = json.loads((vdir / "labels.json").read_text(encoding="utf-8"))
        emb_path = vdir / "train_embeddings.npy"
        train_emb = np.load(emb_path) if emb_path.is_file() else None

        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=embedding_model)
        return cls(
            embedder,
            head,
            labels,
            threshold,
            version,
            embedding_model,
            train_embeddings=train_emb,
            similarity_floor=similarity_floor,
        )

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._embedder.embed([text]))), dtype=np.float32)

    def explain(self, text: str) -> Prediction:
        """Full detail: the decision plus the class ranking and the OOD score."""
        if not text or not text.strip():
            return Prediction(UNKNOWN, 0.0, [], 0.0)
        vec = self.embed(text)
        proba = self._head.predict_proba([vec])[0]
        order = np.argsort(proba)[::-1]
        ranking = [(str(self._head.classes_[i]), float(proba[i])) for i in order]
        top_label, top_conf = ranking[0]

        # out-of-domain guard: MiniLM embeddings are L2-normalised, so a dot
        # product is cosine similarity. Junk text lands far from every training
        # phrase even when the LR head is (wrongly) confident about it.
        similarity = (
            float(np.max(self._train_emb @ vec))
            if self._train_emb is not None
            else 1.0
        )
        if top_conf < self.threshold or similarity < self.similarity_floor:
            return Prediction(UNKNOWN, top_conf, ranking, similarity)
        return Prediction(top_label, top_conf, ranking, similarity)

    def predict(self, text: str) -> tuple[str, float]:
        p = self.explain(text)
        return (p.label, p.confidence)
