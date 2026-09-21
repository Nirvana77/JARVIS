"""M2.5: the background half of teach/edit_skill/revert_skill.

A :class:`LearningJob` takes the :class:`~jarvis.factory.flows.LearningRequest`
a finished dialog produced and turns it into a staged candidate:
build (Claude) → validate → *permission decision* → sandbox. Retrain,
self-check, the keep decision and promotion follow in the orchestrator,
which owns `_staged`/`registry`.

The job never speaks or listens. It only calls two injected callables:

- ``notify(text)`` — queue a line to be spoken at the next safe point;
- ``decide(prompt) -> bool`` — queue a yes/no question and wait for its
  answer (the orchestrator asks it at a safe point).

So the session keeps serving commands while this runs — see
PRD/milestone-2.5-background-learning.md.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, Callable

from jarvis.factory import flows as _flows
from jarvis.factory.build import BuildError, build
from jarvis.factory.flows import FlowOutcome, LearningRequest
from jarvis.factory.validate import ValidationError, validate
from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]
Decide = Callable[[str], Awaitable[bool]]

_BASE_PERMISSIONS = frozenset({"pure", "notify"})


async def run_detached(fn, *args):
    """Run blocking ``fn(*args)`` on a daemon thread and await its result.

    Unlike ``asyncio.to_thread`` the thread isn't in the loop's default
    executor, so cancelling a job (shutdown) never has to wait for a
    minute-long Claude call to come back — the thread just finishes, or
    dies with the process."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def deliver(setter, value):
        if not future.done():
            setter(value)

    def post(setter, value):
        try:
            loop.call_soon_threadsafe(deliver, setter, value)
        except RuntimeError:
            pass  # the loop closed (shutdown) while this was still running

    def target():
        try:
            result = fn(*args)
        except BaseException as exc:  # noqa: BLE001 — handed to the awaiting coroutine
            post(future.set_exception, exc)
        else:
            post(future.set_result, result)

    threading.Thread(target=target, name=f"learn:{getattr(fn, '__name__', 'job')}", daemon=True).start()
    return await future


def _sample_params(manifest: SkillManifest) -> dict:
    defaults = {"integer": 1, "number": 1.0, "boolean": True}
    return {
        key: defaults.get((spec or {}).get("type", "string"), "test")
        for key, spec in manifest.params.items()
    }


class LearningJob:
    def __init__(
        self,
        request: LearningRequest,
        *,
        notify: Notify,
        decide: Decide,
        registry,
        generate,
        sandbox,
    ) -> None:
        self.request = request
        self.notify = notify
        self.decide = decide
        #: the latest registry when the job started (staged-but-unmerged
        #: skills included) — what a new name must not collide with
        self.registry = registry
        self.generate = generate
        self.sandbox = sandbox

    async def run(self) -> FlowOutcome:
        request = self.request
        if request.versioning == "revert":
            # already-approved archived code: nothing to generate or sandbox
            return FlowOutcome(
                accepted=True,
                name=request.name,
                module_source=request.module_source,
                manifest=request.manifest,
                reverted_from_version=request.reverted_from_version,
            )
        return await self._generate_validate_sandbox()

    async def _generate_validate_sandbox(self) -> FlowOutcome:
        request = self.request
        name = request.name
        try:
            generated = await run_detached(
                lambda: build(request.spec, generate=self.generate, existing_source=request.existing_source)
            )
        except BuildError as exc:
            log.warning("build failed for %s: %s", name, exc)
            await self.notify(f"I couldn't put '{name}' together, sir.")
            return FlowOutcome(accepted=False, name=name, reason=str(exc))

        try:
            # `validate()` checks `manifest.name` (whatever Claude actually named
            # it) against `known_names` — not `request.name` — so a model that
            # ignores the requested name still can't collide with an existing skill.
            manifest = validate(
                generated.module_source, frozenset(self.registry.names()),
                allow_name=request.allow_name,
            )
        except ValidationError as exc:
            log.warning("validation failed for %s: %s", name, exc)
            await self.notify(f"What I came up with for '{name}' didn't check out, sir.")
            return FlowOutcome(accepted=False, name=name, reason=str(exc))

        # Generated code never runs — not even in the sandbox — before any
        # permission beyond pure/notify has been granted by voice.
        extra_perms = manifest.permissions - _BASE_PERMISSIONS
        if extra_perms:
            granted = await self.decide(
                f"'{manifest.name}' needs {', '.join(sorted(extra_perms))} access. Allow it, sir?"
            )
            if not granted:
                await self.notify(f"Very well, I won't build '{manifest.name}'.")
                return FlowOutcome(accepted=False, name=manifest.name, reason="permission declined")

        staging = _flows.staging_dir()
        module_path = staging / f"{manifest.name}.py"
        test_path = staging / f"_test_{manifest.name}.py"
        module_path.write_text(generated.module_source, encoding="utf-8")
        test_path.write_text(generated.test_source, encoding="utf-8")
        try:
            test_result = await run_detached(
                self.sandbox.run_tests, module_path, test_path, manifest.permissions
            )
            if not test_result.ok:
                log.warning("sandbox tests failed for %s:\n%s", manifest.name, test_result.stderr)
                await self.notify(
                    f"The tests I wrote for '{manifest.name}' didn't pass, sir. I've set it aside."
                )
                module_path.unlink(missing_ok=True)
                return FlowOutcome(accepted=False, name=manifest.name, reason="sandbox tests failed")

            dry_result = await run_detached(
                self.sandbox.dry_run, module_path, _sample_params(manifest), manifest.permissions
            )
            if not dry_result.ok:
                log.warning("sandbox dry-run failed for %s:\n%s", manifest.name, dry_result.stderr)
                await self.notify(f"'{manifest.name}' didn't run cleanly, sir. I've set it aside.")
                module_path.unlink(missing_ok=True)
                return FlowOutcome(accepted=False, name=manifest.name, reason="sandbox dry-run failed")
        except BaseException:
            # cancelled (shutdown) or crashed mid-sandbox: leave nothing staged
            module_path.unlink(missing_ok=True)
            raise
        finally:
            test_path.unlink(missing_ok=True)

        return FlowOutcome(
            accepted=True,
            name=manifest.name,
            module_source=generated.module_source,
            manifest=manifest,
        )
