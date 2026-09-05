"""The voice dialogs — `teach`, `edit_skill`, `revert_skill`.

Each flow only talks (`ask`/`say`, both async — bound to the orchestrator's
`_ask`/`_speak`), builds/validates/sandboxes a candidate module, and returns a
:class:`FlowOutcome`. **None of them touch `orchestrator._staged` or the
registry** — retraining, the final voice confirm, and promotion are the
orchestrator's `_promote_and_retrain()`, which is the sole writer of that
state (see PRD/milestone-2-skill-factory.md, decision 2).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from jarvis.factory.build import BuildError, build
from jarvis.factory.spec import SkillSpec
from jarvis.factory.validate import ValidationError, validate
from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

Ask = Callable[[str], Awaitable[str]]
Say = Callable[[str], Awaitable[None]]

_YES = {"yes", "yeah", "yep", "sure", "confirm", "affirmative", "correct", "please do"}
_NO = {"no", "nope", "don't", "do not", "negative", "cancel", "never mind", "stop"}


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


def _sample_params(manifest: SkillManifest) -> dict:
    defaults = {"integer": 1, "number": 1.0, "boolean": True}
    return {
        key: defaults.get((spec or {}).get("type", "string"), "test")
        for key, spec in manifest.params.items()
    }


async def _match_skill_name(
    ask: Ask, say: Say, registry, prompt: str, *, origin: str
) -> str | None:
    candidates = [m.name for m in registry.manifests() if m.origin == origin]
    if not candidates:
        await say("I don't have any skills I can do that with yet, sir.")
        return None
    reply = (await ask(prompt)).strip().lower()
    for name in candidates:
        if name in reply or name.replace("_", " ") in reply:
            return name
    await say("I couldn't tell which skill you meant, sir.")
    return None


async def _generate_validate_sandbox(
    spec: SkillSpec,
    *,
    ask: Ask,
    say: Say,
    registry,
    generate,
    sandbox,
    existing_source: str | None,
    allow_name: str | None,
) -> FlowOutcome:
    await say(f"One moment, sir, let me work on '{spec.name}'.")
    try:
        generated = await asyncio.to_thread(
            build, spec, generate=generate, existing_source=existing_source
        )
    except BuildError as exc:
        log.warning("build failed for %s: %s", spec.name, exc)
        await say("I couldn't put that together, sir.")
        return FlowOutcome(accepted=False, reason=str(exc))

    try:
        # `validate()` checks `manifest.name` (whatever Claude actually named
        # it) against `known_names` — not `spec.name` — so a model that
        # ignores the requested name still can't collide with an existing skill.
        manifest = validate(
            generated.module_source, frozenset(registry.names()), allow_name=allow_name
        )
    except ValidationError as exc:
        log.warning("validation failed for %s: %s", spec.name, exc)
        await say("What I came up with didn't check out, sir.")
        return FlowOutcome(accepted=False, reason=str(exc))

    extra_perms = manifest.permissions - {"pure", "notify"}
    if extra_perms:
        granted = await ask_yes_no(
            ask,
            f"This skill needs {', '.join(sorted(extra_perms))} access. Allow it, sir?",
        )
        if not granted:
            await say("Very well, I won't build that.")
            return FlowOutcome(accepted=False, reason="permission declined")

    module_path = staging_dir() / f"{manifest.name}.py"
    test_path = staging_dir() / f"_test_{manifest.name}.py"
    module_path.write_text(generated.module_source, encoding="utf-8")
    test_path.write_text(generated.test_source, encoding="utf-8")
    try:
        test_result = await asyncio.to_thread(
            sandbox.run_tests, module_path, test_path, manifest.permissions
        )
        if not test_result.ok:
            log.warning("sandbox tests failed for %s:\n%s", manifest.name, test_result.stderr)
            await say("The tests I wrote for that didn't pass, sir. I'll set it aside.")
            module_path.unlink(missing_ok=True)
            return FlowOutcome(accepted=False, reason="sandbox tests failed")

        dry_result = await asyncio.to_thread(
            sandbox.dry_run, module_path, _sample_params(manifest), manifest.permissions
        )
        if not dry_result.ok:
            log.warning("sandbox dry-run failed for %s:\n%s", manifest.name, dry_result.stderr)
            await say("That didn't run cleanly, sir. I'll set it aside.")
            module_path.unlink(missing_ok=True)
            return FlowOutcome(accepted=False, reason="sandbox dry-run failed")
    finally:
        test_path.unlink(missing_ok=True)

    return FlowOutcome(
        accepted=True,
        name=manifest.name,
        module_source=generated.module_source,
        manifest=manifest,
    )


class TeachFlow:
    def __init__(
        self,
        *,
        ask: Ask,
        say: Say,
        registry,
        generate,
        sandbox,
        seed_description: str | None = None,
    ) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.generate = generate
        self.sandbox = sandbox
        self.seed_description = seed_description

    async def run(self) -> FlowOutcome:
        name = await self._ask_name()
        if name is None:
            await self.say("Very well, sir.")
            return FlowOutcome(accepted=False, reason="no usable name")

        description = await self._ask_description()
        if not description:
            await self.say("Never mind, then.")
            return FlowOutcome(accepted=False, reason="no description given")

        extra = (await self.ask("Give me another way you might ask me to do that.")).strip()
        examples = [description] + ([extra] if extra else [])

        return await _generate_validate_sandbox(
            SkillSpec(name=name, description=description, examples=examples),
            ask=self.ask,
            say=self.say,
            registry=self.registry,
            generate=self.generate,
            sandbox=self.sandbox,
            existing_source=None,
            allow_name=None,
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
    def __init__(self, *, ask: Ask, say: Say, registry, generate, sandbox) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.generate = generate
        self.sandbox = sandbox

    async def run(self) -> FlowOutcome:
        name = await _match_skill_name(
            self.ask, self.say, self.registry,
            "Which skill would you like to edit, sir?", origin="learned",
        )
        if name is None:
            return FlowOutcome(accepted=False, reason="no matching skill")

        change = (await self.ask(f"What would you like to change about '{name}'?")).strip()
        if not change:
            await self.say("Never mind, then.")
            return FlowOutcome(accepted=False, reason="no change given")

        current = self.registry.manifest(name)
        try:
            existing_source = learned_source_path(name).read_text(encoding="utf-8")
        except OSError as exc:
            log.error("could not read learned source for %s: %s", name, exc)
            await self.say(f"I couldn't find my own source for '{name}', sir.")
            return FlowOutcome(accepted=False, reason="missing source")

        return await _generate_validate_sandbox(
            SkillSpec(
                name=name,
                description=change,
                examples=current.examples,
                based_on_version=current.version,
            ),
            ask=self.ask,
            say=self.say,
            registry=self.registry,
            generate=self.generate,
            sandbox=self.sandbox,
            existing_source=existing_source,
            allow_name=name,
        )


class RevertSkillFlow:
    def __init__(self, *, ask: Ask, say: Say, registry, config) -> None:
        self.ask = ask
        self.say = say
        self.registry = registry
        self.config = config

    async def run(self) -> FlowOutcome:
        name = await _match_skill_name(
            self.ask, self.say, self.registry,
            "Which skill would you like to revert, sir?", origin="learned",
        )
        if name is None:
            return FlowOutcome(accepted=False, reason="no matching skill")

        versions_dir = self.config.skill_versions_dir(name)
        versions = sorted(
            (int(p.stem[1:]) for p in versions_dir.glob("v*.py") if p.stem[1:].isdigit()),
            reverse=True,
        ) if versions_dir.is_dir() else []
        if not versions:
            await self.say(f"I don't have an earlier version of '{name}' to revert to, sir.")
            return FlowOutcome(accepted=False, reason="no earlier version")

        target = versions[0]
        source = (versions_dir / f"v{target}.py").read_text(encoding="utf-8")
        try:
            manifest = validate(source, frozenset(), allow_name=name)
        except ValidationError as exc:
            # shouldn't happen — this source shipped once already — but don't
            # promote something that no longer checks out
            log.error("archived version v%d of %s no longer validates: %s", target, name, exc)
            await self.say(f"That earlier version of '{name}' doesn't check out anymore, sir.")
            return FlowOutcome(accepted=False, reason="archived version invalid")

        return FlowOutcome(
            accepted=True,
            name=name,
            module_source=source,
            manifest=manifest,
            reverted_from_version=target,
        )
