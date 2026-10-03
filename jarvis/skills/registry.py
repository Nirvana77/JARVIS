"""Skill discovery and dispatch.

Replaces the dynamic ``importlib.import_module(f'actions.{action}')`` in
``command_helper``. Each module under ``jarvis.skills.builtin`` (and, since M2,
``jarvis.skills.learned``) that exposes a ``MANIFEST`` and a ``run`` is
registered under ``MANIFEST.name``; the intent label is that name. So is every
tool an edge has declared (``jarvis.skills.edge``, origin ``edge``). ``dispatch``
builds the :class:`Context` and calls ``run``.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import pkgutil
import shutil
import sys
from pathlib import Path
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
        edges=None,
        memory=None,
        knowledge=None,
    ) -> None:
        self.config = config
        self.reasoner = reasoner
        self.say = say or (lambda text: print(f"Jarvis: {text}"))
        #: `EdgeControl` under `serve`, else None — handed to skills as ctx.edges
        self.edges = edges
        #: M4.5: the `Memory` the orchestrator keeps — a skill gets the view
        #: for the device whose turn it is, as ctx.memory
        self.memory = memory
        #: M5: the knowledge base, or None when it is off — ctx.knowledge
        self.knowledge = knowledge
        self._skills: dict[str, object] = {}

    # -- discovery ------------------------------------------------------------

    @staticmethod
    def _point_learned_package(config) -> None:
        """``jarvis.skills.learned`` is imported from the checkout, but its
        modules are looked up in ``config.learned_skills_dir`` (known issue
        #19). Module names stay ``jarvis.skills.learned.<name>``."""
        try:
            pkg = importlib.import_module(LEARNED_PACKAGE)
        except ModuleNotFoundError:
            return
        pkg.__path__ = [str(config.learned_skills_dir)]

    @classmethod
    def discover(
        cls,
        config,
        reasoner=None,
        say: Callable[[str], None] | None = None,
        packages: tuple[str, ...] = DEFAULT_PACKAGES,
        edges=None,
        memory=None,
        knowledge=None,
    ) -> "Registry":
        reg = cls(config, reasoner, say, edges, memory, knowledge)
        if LEARNED_PACKAGE in packages:
            cls._point_learned_package(config)
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
        # What the edges said they can do (jarvis/skills/edge.py), kept on
        # disk, so they are skills — and trained — even while the edge is away.
        from jarvis.skills.edge import EdgeTools, edge_skills

        store = EdgeTools(config.remote_dir)
        for skill in edge_skills(store, reserved=frozenset(reg._skills)):
            reg._skills[skill.MANIFEST.name] = skill
        log.info("registered skills: %s", ", ".join(sorted(reg._skills)))
        return reg

    def rebuilt(self) -> "Registry":
        """M2: a fresh registry that also picks up newly-promoted
        skills/learned/* modules (re-imports everything from scratch)."""
        return Registry.discover(
            self.config, self.reasoner, self.say, edges=self.edges,
            memory=self.memory, knowledge=self.knowledge,
        )

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
            edges=self.edges,
            memory=self.memory.device() if self.memory is not None else None,
            knowledge=self.knowledge,
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


#: where learned skills were kept before known issue #19: the checkout (dev
#: brain), or the PVC the pod mounts over it
CHECKOUT_LEARNED_DIR = Path(__file__).resolve().parent / "learned"


def migrate_learned_skills(old: Path, new: Path) -> list[str]:
    """Copy the learned skills in ``old`` that ``new`` does not have yet.
    Never overwrites (the shared copy wins), never deletes (a brain still on
    older code may be reading ``old``). Returns the names copied."""
    old, new = Path(old), Path(new)
    if not old.is_dir() or old.resolve() == new.resolve():
        return []
    copied = []
    for path in sorted(old.glob("*.py")):
        if path.name.startswith("_"):
            continue
        target = new / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                log.warning(
                    "learned skill %s differs between %s and %s; keeping the shared one",
                    path.stem, old, new,
                )
            continue
        new.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(path.stem)
    if copied:
        log.info("moved learned skill(s) to %s: %s", new, ", ".join(copied))
    return copied
