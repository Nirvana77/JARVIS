"""Skill discovery and dispatch.

Replaces the dynamic ``importlib.import_module(f'actions.{action}')`` in
``command_helper``. Each module under ``jarvis.skills.builtin`` (and, since M2,
``jarvis.skills.learned``) that exposes a ``MANIFEST`` and a ``run`` is
registered under ``MANIFEST.name``; the intent label is that name. ``dispatch``
builds the :class:`Context` and calls ``run``.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import pkgutil
import sys
from typing import Callable

from jarvis.core.context import Context
from jarvis.skills.contract import SkillError, SkillManifest, SkillNotFound

log = logging.getLogger(__name__)

BUILTIN_PACKAGE = "jarvis.skills.builtin"
LEARNED_PACKAGE = "jarvis.skills.learned"
DEFAULT_PACKAGES = (BUILTIN_PACKAGE, LEARNED_PACKAGE)

#: which `origin` a manifest gets, keyed by the package it was actually found
#: in — never trusted from the module's own `MANIFEST` literal. A generated
#: skill's code has no reason to declare `origin` correctly (nothing in the
#: factory prompt asks it to, and there's no way to enforce it via `validate`
#: on a value that only matters *after* promotion, once the file is
#: re-imported from `skills/learned/`), so this is the one place origin is
#: authoritative.
_ORIGIN_FOR_PACKAGE = {BUILTIN_PACKAGE: "builtin", LEARNED_PACKAGE: "learned"}


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
        packages: tuple[str, ...] = DEFAULT_PACKAGES,
    ) -> "Registry":
        reg = cls(config, reasoner, say)
        for package in packages:
            try:
                pkg = importlib.import_module(package)
            except ModuleNotFoundError:
                continue  # e.g. skills/learned/ not present yet
            for info in pkgutil.iter_modules(pkg.__path__):
                if info.name.startswith("_"):
                    continue
                full_name = f"{package}.{info.name}"
                # A learned skill's module name is stable across edit_skill /
                # revert_skill overwriting its file on disk — reload rather
                # than trust the (stale) sys.modules cache.
                if full_name in sys.modules:
                    mod = importlib.reload(sys.modules[full_name])
                else:
                    mod = importlib.import_module(full_name)
                manifest = getattr(mod, "MANIFEST", None)
                run = getattr(mod, "run", None)
                if not isinstance(manifest, SkillManifest) or not callable(run):
                    log.warning("skipping %s: missing MANIFEST or run()", info.name)
                    continue
                expected_origin = _ORIGIN_FOR_PACKAGE.get(package)
                if expected_origin is not None and manifest.origin != expected_origin:
                    manifest = dataclasses.replace(manifest, origin=expected_origin)
                    mod.MANIFEST = manifest  # keep `mod.MANIFEST` (read elsewhere) in sync
                if manifest.name in reg._skills:
                    log.warning("duplicate skill name %r (%s)", manifest.name, info.name)
                reg._skills[manifest.name] = mod
        log.info("registered skills: %s", ", ".join(sorted(reg._skills)))
        return reg

    def rebuilt(self) -> "Registry":
        """M2: a fresh registry that also picks up newly-promoted
        skills/learned/* modules (re-imports everything from scratch)."""
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
