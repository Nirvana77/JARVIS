"""Pure orchestration between a `SkillSpec` and a generated module.

No network here — `generate` is injected (default: a bound
`ClaudeClient.generate_skill`, via `default_generate`), the same DI pattern
used throughout `jarvis/core` for `wake`/`mic`/`stt`/`tts`/`nlu`/`reasoner`.
Tests pass a fake `generate` and never touch the network.
"""

from __future__ import annotations

from typing import Callable, Protocol

from jarvis.factory.spec import SkillSpec


class GeneratedSkillLike(Protocol):
    name: str
    module_source: str
    test_source: str


Generate = Callable[[SkillSpec, "str | None", "str | None"], GeneratedSkillLike]


class BuildError(RuntimeError):
    """`generate` failed, or returned something obviously unusable."""


def build(
    spec: SkillSpec,
    *,
    generate: Generate,
    existing_source: str | None = None,
    feedback: str | None = None,
) -> GeneratedSkillLike:
    """`feedback`, when given, describes why a *previous* attempt at this same
    spec didn't check out (a validation error or sandbox failure) — passed
    straight through to `generate` so a retry can ask Claude to fix that
    specific problem instead of generating blind again."""
    try:
        generated = generate(spec, existing_source, feedback)
    except Exception as exc:  # noqa: BLE001 — degrade to a spoken failure, not a crash
        raise BuildError(f"couldn't generate '{spec.name}': {exc}") from exc

    if not generated.module_source.strip() or not generated.test_source.strip():
        raise BuildError(f"'{spec.name}': generated module or test source was empty")
    return generated
