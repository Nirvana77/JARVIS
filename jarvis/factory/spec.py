"""The clarified request a `teach`/`edit_skill` dialog hands to the factory.

Deliberately thin — the dialog (`flows.py`) only gathers a name, a plain-
language description, and a couple of example phrasings by voice. Claude
infers the rest (params, permissions) as part of generating the module;
`validate.py` is what actually enforces the permission vocabulary, not this
spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    #: only set for edit_skill — the manifest.version the edit is based on
    based_on_version: int | None = None
