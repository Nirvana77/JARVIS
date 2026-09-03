"""Build the NLU training corpus from the hand-authored seed + skill manifests.

``intents.json`` stays the human-editable seed (builtin + meta intents,
persona-critical canned lines); the trainer never writes to it. Each skill
module owns its own ``examples`` in its ``MANIFEST``. The corpus the trainer
actually reads is ``seed patterns`` + ``every registered skill's examples``,
cached in ``data/nlu/corpus.sqlite`` and rebuildable with ``jarvis nlu rebuild``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from jarvis.skills.contract import SkillManifest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INTENTS_PATH = _REPO_ROOT / "intents.json"

#: concrete fillers substituted for a ``{intent}`` placeholder in a seed pattern.
#: a mix of bare noun phrases and short clauses so the head generalises to both
#: "search black holes" and "note that the meeting moved to friday".
PLACEHOLDER_FILLERS = (
    "black holes",
    "the news",
    "lofi music",
    "cats",
    "python tutorials",
    "the weather in paris",
    "the meeting moved to friday",
    "i parked on level two",
    "buy milk on the way home",
    "the wifi password is on the router",
)


@dataclass(frozen=True)
class SeedIntent:
    tag: str
    patterns: list[str]
    responses: list[str]
    action: str


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    source: str  # "seed" or "skill"


def load_seed_intents(path: str | Path = DEFAULT_INTENTS_PATH) -> list[SeedIntent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intents: list[SeedIntent] = []
    for entry in data["intents"]:
        intents.append(
            SeedIntent(
                tag=entry["tag"],
                patterns=list(entry.get("patterns", [])),
                responses=list(entry.get("responses", [])),
                action=entry.get("action", "none"),
            )
        )
    return intents


def intent_meta(
    path: str | Path = DEFAULT_INTENTS_PATH,
) -> dict[str, SeedIntent]:
    """``tag -> SeedIntent`` — used by the orchestrator for actions/responses."""
    return {si.tag: si for si in load_seed_intents(path)}


def expand_pattern(pattern: str) -> list[str]:
    """A ``{intent}`` placeholder becomes several concrete phrasings."""
    if "{intent}" not in pattern:
        return [pattern]
    out = [pattern.replace("{intent}", filler) for filler in PLACEHOLDER_FILLERS]
    stripped = " ".join(pattern.replace("{intent}", " ").split())
    if stripped:
        out.append(stripped)
    return out


def build_corpus(
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    manifests: Sequence[SkillManifest] = (),
) -> list[Example]:
    seen: set[tuple[str, str]] = set()
    examples: list[Example] = []

    def add(text: str, label: str, source: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        key = (text.lower(), label)
        if key in seen:
            return
        seen.add(key)
        examples.append(Example(text=text, label=label, source=source))

    for si in load_seed_intents(intents_path):
        for pattern in si.patterns:
            for phrasing in expand_pattern(pattern):
                add(phrasing, si.tag, "seed")

    for manifest in manifests:
        for example in manifest.examples:
            add(example, manifest.name, "skill")

    return examples


# -- sqlite cache ---------------------------------------------------------------

def write_corpus_db(examples: Iterable[Example], db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS examples")
        conn.execute(
            "CREATE TABLE examples (text TEXT NOT NULL, label TEXT NOT NULL, "
            "source TEXT NOT NULL, UNIQUE(text, label))"
        )
        conn.executemany(
            "INSERT OR IGNORE INTO examples (text, label, source) VALUES (?, ?, ?)",
            [(e.text, e.label, e.source) for e in examples],
        )


def read_corpus_db(db_path: str | Path) -> list[Example]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT text, label, source FROM examples ORDER BY label, text"
        ).fetchall()
    return [Example(text=t, label=l, source=s) for t, l, s in rows]
