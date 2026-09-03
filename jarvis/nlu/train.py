"""Train the NLU head: embed the corpus, fit a sklearn classifier, version it.

Replaces ``libs/training.py`` (bag-of-words + Keras). The head is a
``LogisticRegression`` over ``fastembed`` MiniLM embeddings. Artifacts land in
``data/models/nlu/v<N>/`` and the last 3 versions are kept for rollback (M2).

M1 runs this in-process. The worker-process / container retrain and the idle
merge gate are M2.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from jarvis.nlu.corpus import Example

KEEP_VERSIONS = 3

# Embeddings from fastembed MiniLM are already L2-normalised and the seed corpus
# is small (a few hundred short phrases); the default C=1 leaves the softmax
# probabilities too flat to threshold usefully, so loosen the regularisation.
_LR_C = 10.0


@dataclass(frozen=True)
class TrainResult:
    version: int
    path: Path
    n_examples: int
    n_labels: int
    accuracy: float | None


def _embed(texts: Sequence[str], embedding_model: str) -> np.ndarray:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=embedding_model)
    return np.asarray(list(embedder.embed(list(texts))), dtype=np.float32)


def _next_version(out_dir: Path) -> int:
    existing = [
        int(p.name[1:])
        for p in out_dir.glob("v*")
        if p.is_dir() and p.name[1:].isdigit()
    ]
    return (max(existing) + 1) if existing else 1


def latest_version(out_dir: Path) -> int | None:
    versions = [
        int(p.name[1:])
        for p in Path(out_dir).glob("v*")
        if p.is_dir() and p.name[1:].isdigit() and (p / "head.joblib").is_file()
    ]
    return max(versions) if versions else None


def _prune(out_dir: Path, keep: int = KEEP_VERSIONS) -> None:
    versions = sorted(
        (int(p.name[1:]) for p in out_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit()),
        reverse=True,
    )
    for stale in versions[keep:]:
        shutil.rmtree(out_dir / f"v{stale}", ignore_errors=True)


def _estimate_accuracy(X: np.ndarray, y: np.ndarray) -> float | None:
    labels, counts = np.unique(y, return_counts=True)
    # need at least 2 per class and >1 class to hold out a stratified split
    if len(labels) < 2 or counts.min() < 2 or len(y) < 10:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    probe = LogisticRegression(C=_LR_C, max_iter=1000)
    probe.fit(X_tr, y_tr)
    return round(float(probe.score(X_te, y_te)), 4)


def train(
    examples: Sequence[Example],
    embedding_model: str,
    out_dir: str | Path,
) -> TrainResult:
    if not examples:
        raise ValueError("cannot train on an empty corpus")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [e.text for e in examples]
    y = np.array([e.label for e in examples])
    X = _embed(texts, embedding_model)

    accuracy = _estimate_accuracy(X, y)

    head = LogisticRegression(C=_LR_C, max_iter=1000)
    head.fit(X, y)

    version = _next_version(out_dir)
    vdir = out_dir / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(head, vdir / "head.joblib")
    # keep the training embedding matrix so inference can reject out-of-domain
    # text by nearest-neighbour similarity (LR alone is overconfident on junk)
    np.save(vdir / "train_embeddings.npy", X)
    (vdir / "labels.json").write_text(
        json.dumps(head.classes_.tolist(), indent=2), encoding="utf-8"
    )
    (vdir / "meta.json").write_text(
        json.dumps(
            {
                "version": version,
                "embedding_model": embedding_model,
                "n_examples": len(examples),
                "labels": head.classes_.tolist(),
                "accuracy": accuracy,
                "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _prune(out_dir)

    return TrainResult(
        version=version,
        path=vdir,
        n_examples=len(examples),
        n_labels=int(len(head.classes_)),
        accuracy=accuracy,
    )
