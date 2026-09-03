"""The skill contract — see PRD/jarvis-2026-rebuild.md § "The Skill contract".

A skill is a module under ``jarvis/skills/builtin/`` (M1) or, later,
``jarvis/skills/learned/`` (M2). It exposes a module-level ``MANIFEST`` and a
``run(ctx, **params) -> str`` that returns the line to speak. A skill never
imports the audio/core stack — it only touches ``ctx`` — which is what breaks
the old ``actions/write.py`` <-> ``command_helper`` import cycle.

``permissions`` are *declared* in M1 but not enforced; the AST allowlist and
sandbox that enforce them arrive in M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid importing Context at module load (keeps skills light)
    from jarvis.core.context import Context

#: the closed vocabulary a manifest's ``permissions`` may draw from
PERMISSIONS = frozenset(
    {"pure", "notify", "fs_read", "fs_write", "net", "shell"}
)


class SkillNotFound(KeyError):
    """No registered skill matches the requested intent label."""


class SkillError(RuntimeError):
    """A skill's ``run()`` raised or returned something unusable."""


@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    #: JSON-schema-ish param spec, e.g. {"query": {"type": "string", "required": True}}
    params: dict = field(default_factory=dict)
    permissions: frozenset[str] = frozenset({"pure"})
    version: int = 1
    origin: str = "builtin"  # or "learned"

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"skill name must be an identifier, got {self.name!r}")
        unknown = set(self.permissions) - PERMISSIONS
        if unknown:
            raise ValueError(
                f"{self.name}: unknown permission(s) {sorted(unknown)}; "
                f"allowed: {sorted(PERMISSIONS)}"
            )

    @property
    def required_params(self) -> list[str]:
        return [
            key
            for key, spec in self.params.items()
            if isinstance(spec, dict) and spec.get("required")
        ]


@runtime_checkable
class Skill(Protocol):
    MANIFEST: SkillManifest

    def run(self, ctx: "Context", **params: object) -> str: ...
