"""Skill discovery and dispatch.

Replaces the dynamic ``importlib.import_module(f'actions.{action}')`` in
``command_helper``. Each module under ``jarvis.skills.builtin`` that exposes a
``MANIFEST`` and a ``run`` is registered under ``MANIFEST.name``; the intent
label is that name. ``dispatch`` builds the :class:`Context` and calls ``run``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable

from jarvis.core.context import Context
from jarvis.skills.contract import SkillError, SkillManifest, SkillNotFound

log = logging.getLogger(__name__)

BUILTIN_PACKAGE = "jarvis.skills.builtin"


class Registry:
    def __init__(
        self,
        config,
        reasoner=None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.reasoner = reasoner
        self.say = say or (lambda text: print(f"Jarvis: {text}"))
        self._skills: dict[str, object] = {}

    # -- discovery ------------------------------------------------------------

    @classmethod
    def discover(
        cls,
        config,
        reasoner=None,
        say: Callable[[str], None] | None = None,
        package: str = BUILTIN_PACKAGE,
    ) -> "Registry":
        reg = cls(config, reasoner, say)
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            mod = importlib.import_module(f"{package}.{info.name}")
            manifest = getattr(mod, "MANIFEST", None)
            run = getattr(mod, "run", None)
            if not isinstance(manifest, SkillManifest) or not callable(run):
                log.warning("skipping %s: missing MANIFEST or run()", info.name)
                continue
            if manifest.name in reg._skills:
                log.warning("duplicate skill name %r (%s)", manifest.name, info.name)
            reg._skills[manifest.name] = mod
        log.info("registered skills: %s", ", ".join(sorted(reg._skills)))
        return reg

    def rebuilt(self) -> "Registry":
        """M2 seam: a fresh registry that also picks up skills/learned/*."""
        return Registry.discover(self.config, self.reasoner, self.say)

    # -- introspection -----------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._skills)

    def manifest(self, name: str) -> SkillManifest:
        return self._skills[name].MANIFEST

    def manifests(self) -> list[SkillManifest]:
        return [m.MANIFEST for m in self._skills.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    # -- dispatch -----------------------------------------------------------

    def _context(self, name: str) -> Context:
        return Context(
            say=self.say,
            config=self.config,
            _data_dir=self.config.skill_data_dir(name),
            llm=self.reasoner,
        )

    def dispatch(self, label: str, params: dict | None = None) -> str:
        mod = self._skills.get(label)
        if mod is None:
            raise SkillNotFound(label)
        declared = set(mod.MANIFEST.params)
        params = {
            k: v
            for k, v in (params or {}).items()
            if v != "" and (not declared or k in declared)
        }
        try:
            result = mod.run(self._context(label), **params)
        except TypeError as exc:  # bad slot wiring
            raise SkillError(f"{label}: {exc}") from exc
        if not isinstance(result, str):
            raise SkillError(f"{label}.run returned {type(result).__name__}, expected str")
        return result
