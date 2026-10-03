"""M2.5: the background half of teach/edit_skill/revert_skill/remove_skill.

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

#: default cap for `LearningJob`'s generate/validate/sandbox retry loop —
#: overridable per-job (the orchestrator passes
#: `config.factory.max_generate_attempts`); tests that don't care leave it
#: at the default.
DEFAULT_MAX_ATTEMPTS = 5


async def run_detached(fn, *args, name: str | None = None):
    """Run blocking ``fn(*args)`` on a daemon thread and await its result.

    Unlike ``asyncio.to_thread`` the thread isn't in the loop's default
    executor, so cancelling a job (shutdown) never has to wait for a
    minute-long Claude call to come back — the thread just finishes, or
    dies with the process. ``name`` is the thread's name, for a caller that
    isn't a learning job."""
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

    name = name or f"learn:{getattr(fn, '__name__', 'job')}"
    threading.Thread(target=target, name=name, daemon=True).start()
    return await future


def _sandbox_output(result) -> str:
    """What a sandbox run printed. pytest reports failures on *stdout* and
    tracebacks land on stderr, so both are needed — stderr alone left the
    log and Claude's retry feedback empty for every failing test."""
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


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
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.request = request
        self.notify = notify
        self.decide = decide
        #: the latest registry when the job started (staged-but-unmerged
        #: skills included) — what a new name must not collide with
        self.registry = registry
        self.generate = generate
        self.sandbox = sandbox
        self.max_attempts = max_attempts

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
        if request.versioning == "remove":
            # tearing down already-approved code: nothing to generate, sandbox,
            # or write — the orchestrator's promotion step does the quarantine
            return FlowOutcome(accepted=True, name=request.name, manifest=request.manifest)
        return await self._generate_validate_sandbox()

    async def _generate_validate_sandbox(self) -> FlowOutcome:
        """Generate → validate → sandbox, retrying up to `self.max_attempts`
        times. A validation or sandbox failure doesn't end the job outright —
        the error (validation message, or sandbox stderr) is handed back to
        `generate` as `feedback` so the next attempt can actually fix the
        problem instead of generating blind again. Only a declined permission
        grant ends the job immediately (that's the user's call, not something
        a retry can fix). The retries in between are silent (logged, not
        notified) so a fixable hiccup doesn't turn into a string of "didn't
        work" notices; only the final give-up (if every attempt is exhausted)
        reaches the user."""
        request = self.request
        name = request.name
        feedback: str | None = None
        granted_perms: set[str] = set()
        reason = "unknown error"

        for attempt in range(1, self.max_attempts + 1):
            try:
                generated = await run_detached(
                    lambda: build(
                        request.spec, generate=self.generate,
                        existing_source=request.existing_source, feedback=feedback,
                    )
                )
            except BuildError as exc:
                log.warning(
                    "build failed for %s (attempt %d/%d): %s", name, attempt, self.max_attempts, exc
                )
                reason = str(exc)
                feedback = f"The last attempt failed outright: {exc}"
                continue

            try:
                # `validate()` checks `manifest.name` (whatever Claude actually
                # named it) against `known_names` — not `request.name` — so a
                # model that ignores the requested name still can't collide
                # with an existing skill.
                manifest = validate(
                    generated.module_source, frozenset(self.registry.names()),
                    allow_name=request.allow_name,
                )
            except ValidationError as exc:
                log.warning(
                    "validation failed for %s (attempt %d/%d): %s",
                    name, attempt, self.max_attempts, exc,
                )
                reason = str(exc)
                feedback = (
                    f"Your previous attempt failed validation: {exc}\n\n"
                    f"Here is the code you wrote — fix it:\n```python\n{generated.module_source}\n```"
                )
                continue

            # Generated code never runs — not even in the sandbox — before any
            # permission beyond pure/notify has been granted by voice. A
            # permission already granted on an earlier attempt isn't re-asked
            # if a retry needs that same one again.
            extra_perms = manifest.permissions - _BASE_PERMISSIONS
            if extra_perms and not extra_perms <= granted_perms:
                granted = await self.decide(
                    f"'{manifest.name}' needs {', '.join(sorted(extra_perms))} access. Allow it, sir?"
                )
                if not granted:
                    await self.notify(f"Very well, I won't build '{manifest.name}'.")
                    return FlowOutcome(accepted=False, name=manifest.name, reason="permission declined")
                granted_perms |= extra_perms

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
                    log.warning(
                        "sandbox tests failed for %s (attempt %d/%d):\n%s",
                        manifest.name, attempt, self.max_attempts, _sandbox_output(test_result),
                    )
                    reason = "sandbox tests failed"
                    feedback = (
                        "Your previous attempt's tests failed when run in the sandbox. "
                        "Fix the skill (or the test, if the test itself is wrong) so the "
                        f"tests pass. Error output:\n{_sandbox_output(test_result)}\n\n"
                        f"Here is the code you wrote:\n```python\n{generated.module_source}\n```\n\n"
                        f"Here is the test you wrote:\n```python\n{generated.test_source}\n```"
                    )
                    module_path.unlink(missing_ok=True)
                    continue

                dry_result = await run_detached(
                    self.sandbox.dry_run, module_path, _sample_params(manifest), manifest.permissions
                )
                if not dry_result.ok:
                    log.warning(
                        "sandbox dry-run failed for %s (attempt %d/%d):\n%s",
                        manifest.name, attempt, self.max_attempts, _sandbox_output(dry_result),
                    )
                    reason = "sandbox dry-run failed"
                    feedback = (
                        "Your previous attempt's tests passed, but calling run() for real "
                        f"crashed. Error output:\n{_sandbox_output(dry_result)}\n\n"
                        f"Here is the code you wrote — fix it:\n```python\n{generated.module_source}\n```"
                    )
                    module_path.unlink(missing_ok=True)
                    continue

                if request.replay_params is not None:
                    # M7 repair: the call that failed in use must work now
                    replay = await run_detached(
                        self.sandbox.dry_run, module_path, request.replay_params,
                        manifest.permissions,
                    )
                    if not replay.ok:
                        log.warning(
                            "repair of %s still fails the call that broke it (attempt %d/%d):\n%s",
                            manifest.name, attempt, self.max_attempts, _sandbox_output(replay),
                        )
                        reason = "the failing call still fails"
                        feedback = (
                            "Your repair passes its tests, but the call that failed in use "
                            f"still fails: run(ctx, **{request.replay_params!r}). Error output:\n"
                            f"{_sandbox_output(replay)}\n\n"
                            f"Here is the code you wrote — fix it:\n```python\n{generated.module_source}\n```"
                        )
                        module_path.unlink(missing_ok=True)
                        continue
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

        log.warning(
            "gave up on %s after %d attempts, last reason: %s", name, self.max_attempts, reason
        )
        await self.notify(
            f"I tried a few different ways with '{name}', sir, but couldn't get it working. "
            "I've set it aside."
        )
        return FlowOutcome(accepted=False, name=name, reason=reason)
