"""Edge tools: what an edge says it can do, as skills JARVIS can call.

An edge lists its tools in ``hello`` (``protocol.validate_tools``): a name, a
description, example phrases, and typed params. The brain keeps the list per
device in ``data/remote/tools/<device_id>.json``, so it outlives the
connection, and the registry turns every tool into an ``edge`` skill
(``edge_skills``). The classifier learns a tool from its examples like any
other skill; a tool without examples (``notify``) is never routed by voice,
only called — by ``RemoteServer.notify``.

Running an edge skill sends ``call`` to the edge (``ctx.edges.call``) and
speaks what its ``result`` says. While the edge is away the skill still
exists, and says so: "The watch isn't connected."
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

#: what a missing required param of each type asks
ASK = {
    "duration": "How long, sir?",
    "number": "What number, sir?",
    "text": "What should it say, sir?",
}


class EdgeTools:
    """``data/remote/tools/<device_id>.json``: the tools each edge declared."""

    def __init__(self, remote_dir: str | Path) -> None:
        self.dir = Path(remote_dir) / "tools"

    def _path(self, device_id: str) -> Path:
        # device_id is validated by the protocol (it is a filename already
        # for the addressing mode).
        return self.dir / f"{device_id}.json"

    def tools(self, device_id: str) -> list[dict]:
        try:
            return json.loads(self._path(device_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def devices(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def save(self, device_id: str, tools: list[dict]) -> bool:
        """Keep the list; ``True`` when it differs from the one kept before
        (new firmware with new tools: time to retrain)."""
        if self.tools(device_id) == tools:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(device_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(tools, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        log.info("edge %s declares %d tool(s): %s", device_id, len(tools),
                 ", ".join(t["name"] for t in tools))
        return True


def missing(manifest: SkillManifest, params: dict) -> list[tuple[str, dict, str]]:
    """The required params ``params`` lacks: ``(name, spec, question)`` each,
    for the orchestrator to ask before it runs the skill."""
    return [
        (name, manifest.params[name], ASK.get(manifest.params[name].get("type"), ASK["text"]))
        for name in manifest.required_params
        if name not in params
    ]


@dataclass(frozen=True)
class EdgeSkill:
    """A skill whose ``run`` is a ``call`` to the edge. Quacks like a skill
    module (``MANIFEST`` + ``run``), which is all the registry asks."""

    MANIFEST: SkillManifest
    device_id: str

    def run(self, ctx, **params) -> str:
        if ctx.edges is None:
            return "The watch isn't connected to me."
        for name in self.MANIFEST.required_params:
            if name not in params:
                kind = self.MANIFEST.params[name].get("type", "text")
                return ASK.get(kind, "I need a little more to go on, sir.")
        result = ctx.edges.call(self.MANIFEST.name, params)
        if result.status == "offline":
            return "The watch isn't connected to me."
        if result.status == "timeout":
            return "The watch didn't answer, sir."
        if result.status == "unsupported":
            return "The watch can't do that any more, sir."
        if result.say:
            return result.say
        return "Done." if result.status == "ok" else "The watch couldn't do that, sir."


def edge_skills(store: EdgeTools, reserved: frozenset[str] = frozenset()) -> list[EdgeSkill]:
    """Every kept tool as a skill. A name in ``reserved`` (a builtin or
    learned skill) is skipped: the edge cannot shadow JARVIS's own skills."""
    skills: dict[str, EdgeSkill] = {}
    for device_id in store.devices():
        for tool in store.tools(device_id):
            name = tool.get("name", "")
            if name in reserved:
                log.warning("edge %s tool %r clashes with a skill; ignored", device_id, name)
                continue
            if name in skills:
                continue  # two devices with the same tool: the first one
            try:
                manifest = SkillManifest(
                    name=name,
                    description=tool.get("description", ""),
                    examples=list(tool.get("examples", [])),
                    params=dict(tool.get("params", {})),
                    permissions=frozenset({"notify"}),
                    origin="edge",
                )
            except ValueError as exc:
                log.warning("edge %s tool %r: %s", device_id, name, exc)
                continue
            skills[name] = EdgeSkill(manifest, device_id)
    return list(skills.values())
