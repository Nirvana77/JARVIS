"""The voice dialogs — `teach`, `edit_skill`, `revert_skill`.

M2.5: a flow is **only the dialog** — the questions only the user can
answer. It returns a :class:`LearningRequest` and is done; building,
validating, sandboxing and retraining happen afterwards in a background
:class:`jarvis.factory.jobs.LearningJob`, so the session keeps serving
commands meanwhile. **None of them touch `orchestrator._staged` or the
registry** — the orchestrator is the sole writer of that state (see
PRD/milestone-2-skill-factory.md, decision 2, and
PRD/milestone-2.5-background-learning.md).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from jarvis.factory.spec import SkillSpec
from jarvis.factory.validate import ValidationError, validate
from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

Ask = Callable[[str], Awaitable[str]]
Say = Callable[[str], Awaitable[None]]

_YES = {"yes", "yeah", "yep", "sure", "confirm", "affirmative", "correct", "please do"}
_NO = {"no", "nope", "don't", "do not", "negative", "cancel", "never mind", "stop"}


Versioning = Literal["new", "edit", "revert"]


@dataclass
class LearningRequest:
    """What a finished dialog hands to the background job.

    ``new``/``edit`` carry a ``spec`` for Claude; ``revert`` carries the
    already-approved archived source to restore (no Claude, no sandbox)."""

    versioning: Versioning
    name: str
    spec: SkillSpec | None = None
    existing_source: str | None = None
    allow_name: str | None = None
    module_source: str | None = None
    manifest: SkillManifest | None = None
    reverted_from_version: int | None = None


@dataclass
class FlowOutcome:
    accepted: bool
    name: str | None = None
    module_source: str | None = None
    manifest: SkillManifest | None = None
    reason: str | None = None
    reverted_from_version: int | None = None


async def ask_yes_no(ask: Ask, prompt: str) -> bool:
    reply = (await ask(prompt)).strip().lower()
    if any(word in reply for word in _NO):
        return False
    return any(word in reply for word in _YES)


def _words(reply: str) -> str:
    return " " + " ".join(re.findall(r"[a-z']+", reply.lower())) + " "


async def ask_yes_no_or_none(ask: Ask, prompt: str) -> bool | None:
    """Like :func:`ask_yes_no`, but anything that is neither a clear yes nor a
    clear no is ``None`` — M2.5's background questions land while the user
    may be thinking about something else, so silence or an unrelated reply
    must not count as "no". Whole-word matching, so "know"/"now" aren't "no"."""
    words = _words(await ask(prompt))
    if any(f" {w} " in words for w in _NO):
        return False
    if any(f" {w} " in words for w in _YES):
        return True
    return None


def staging_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "skills" / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def learned_source_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "skills" / "learned" / f"{name}.py"


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not text:
        return ""
    if text[0].isdigit():
        text = f"_{text}"
    return text if text.isidentifier() else ""


async def _match_skill_name(
    ask: Ask, say: Say, registry, prompt: str, *, origin: str,
    busy_names: frozenset[str] = frozenset(),
) -> str | None:
    candidates = [m.name for m in registry.manifests() if m.origin == origin]
    if not candidates:
        await say("I don't have any skills I can do that with yet, sir.")
        return None
    reply = (await ask(prompt)).strip().lower()
    for name in candidates:
        if name in reply or name.replace("_", " ") in reply:
            if name in busy_names:
                await say(f"I'm already working on '{name}', sir.")
                return None
            return name
    await say("I couldn't tell which skill you meant, sir.")
    return None


class TeachFlow:
    def __init__(
        self,
        *,
        ask: Ask,
        say: Say,
        registry,
        seed_description: str | None = None,
        busy_names: frozenset[str] = frozenset(),
    ) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.seed_description = seed_description
        #: skills a background job is already learning — not re-teachable yet
        self.busy_names = busy_names

    async def run(self) -> LearningRequest | None:
        name = await self._ask_name()
        if name is None:
            await self.say("Very well, sir.")
            return None

        description = await self._ask_description()
        if not description:
            await self.say("Never mind, then.")
            return None

        extra = (await self.ask("Give me another way you might ask me to do that.")).strip()
        examples = [description] + ([extra] if extra else [])

        return LearningRequest(
            versioning="new",
            name=name,
            spec=SkillSpec(name=name, description=description, examples=examples),
        )

    async def _ask_name(self) -> str | None:
        for _ in range(2):
            raw = await self.ask("What should I call this skill?")
            name = _slugify(raw)
            if not name:
                await self.say("I didn't catch a usable name, sir.")
                continue
            if name in self.registry:
                await self.say(f"I already have a skill called '{name}', sir.")
                continue
            if name in self.busy_names:
                await self.say(f"I'm already working on '{name}', sir.")
                continue
            return name
        return None

    async def _ask_description(self) -> str:
        if self.seed_description:
            confirmed = await ask_yes_no(
                self.ask,
                f"So you'd like me to be able to {self.seed_description}. Is that right, sir?",
            )
            if confirmed:
                return self.seed_description
        return (await self.ask("In your own words, what should this skill do?")).strip()


class EditSkillFlow:
    def __init__(
        self, *, ask: Ask, say: Say, registry, busy_names: frozenset[str] = frozenset()
    ) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.busy_names = busy_names

    async def run(self) -> LearningRequest | None:
        name = await _match_skill_name(
            self.ask, self.say, self.registry,
            "Which skill would you like to edit, sir?", origin="learned",
            busy_names=self.busy_names,
        )
        if name is None:
            return None

        change = (await self.ask(f"What would you like to change about '{name}'?")).strip()
        if not change:
            await self.say("Never mind, then.")
            return None

        current = self.registry.manifest(name)
        try:
            existing_source = learned_source_path(name).read_text(encoding="utf-8")
        except OSError as exc:
            log.error("could not read learned source for %s: %s", name, exc)
            await self.say(f"I couldn't find my own source for '{name}', sir.")
            return None

        return LearningRequest(
            versioning="edit",
            name=name,
            spec=SkillSpec(
                name=name,
                description=change,
                examples=current.examples,
                based_on_version=current.version,
            ),
            existing_source=existing_source,
            allow_name=name,
        )


class RevertSkillFlow:
    def __init__(
        self, *, ask: Ask, say: Say, registry, config, busy_names: frozenset[str] = frozenset()
    ) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.config = config
        self.busy_names = busy_names

    async def run(self) -> LearningRequest | None:
        name = await _match_skill_name(
            self.ask, self.say, self.registry,
            "Which skill would you like to revert, sir?", origin="learned",
            busy_names=self.busy_names,
        )
        if name is None:
            return None

        versions_dir = self.config.skill_versions_dir(name)
        versions = sorted(
            (int(p.stem[1:]) for p in versions_dir.glob("v*.py") if p.stem[1:].isdigit()),
            reverse=True,
        ) if versions_dir.is_dir() else []
        if not versions:
            await self.say(f"I don't have an earlier version of '{name}' to revert to, sir.")
            return None

        target = versions[0]
        source = (versions_dir / f"v{target}.py").read_text(encoding="utf-8")
        try:
            manifest = validate(source, frozenset(), allow_name=name)
        except ValidationError as exc:
            # shouldn't happen — this source shipped once already — but don't
            # promote something that no longer checks out
            log.error("archived version v%d of %s no longer validates: %s", target, name, exc)
            await self.say(f"That earlier version of '{name}' doesn't check out anymore, sir.")
            return None

        return LearningRequest(
            versioning="revert",
            name=name,
            module_source=source,
            manifest=manifest,
            reverted_from_version=target,
        )
