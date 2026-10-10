"""How the Kubernetes cluster is doing: nodes that are ready, pods that are not.

Asked as "check the kubernetes status", with nothing to dictate: the skill
reads the cluster's API itself and says what is wrong, or that nothing is.

Two ways in, tried in this order:

- **inside a pod** (the production brain): the pod's own service account, over
  HTTPS to the API server. It needs a role that may list nodes and pods — the
  default service account has none, and then the answer is "I am not permitted
  to read the cluster, sir."
- **anywhere else** (the dev brain): ``kubectl get --raw``, with whatever
  kubeconfig that machine's ``kubectl`` already uses.

Read-only by construction: two GETs, no verb from the utterance.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests

from jarvis.skills.contract import SkillManifest

log = logging.getLogger(__name__)

_SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
_TIMEOUT = 8
#: unhealthy pods named before "and N more"
_MAX_NAMED = 3

MANIFEST = SkillManifest(
    name="cluster_status",
    description="Check the Kubernetes cluster: which nodes are ready and which pods are not healthy.",
    examples=[
        "check the kubernetes status",
        "what is the kubernetes cluster status",
        "check the cluster status",
        "how is the cluster doing",
        "how is kubernetes doing",
        "kubernetes status",
        "are all the nodes up",
        "is anything down in the cluster",
        "are any pods failing",
        "check the kube status",
        "how is k3s doing",
    ],
    params={},
    permissions=frozenset({"net", "shell"}),
    voice="jarvis",
)

_REASONS = {
    "CrashLoopBackOff": "is crash looping",
    "ImagePullBackOff": "cannot pull its image",
    "ErrImagePull": "cannot pull its image",
    "CreateContainerConfigError": "has a broken configuration",
    "OOMKilled": "ran out of memory",
    "Evicted": "was evicted",
}


class ClusterError(RuntimeError):
    """The cluster could not be read. ``kind`` is ``forbidden``, ``unreachable``
    or ``unavailable`` (no service account and no kubectl here)."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(detail or kind)
        self.kind = kind


def _fetch(path: str) -> dict:
    """One GET against the API server, as JSON."""
    token = _SA_DIR / "token"
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    if host and token.is_file():
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        ca = _SA_DIR / "ca.crt"
        try:
            resp = requests.get(
                f"https://{host}:{port}{path}",
                headers={"Authorization": f"Bearer {token.read_text(encoding='utf-8').strip()}"},
                verify=str(ca) if ca.is_file() else True,
                timeout=_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ClusterError("unreachable", str(exc)) from exc
        if resp.status_code in (401, 403):
            raise ClusterError("forbidden", f"{resp.status_code} for {path}")
        if resp.status_code != 200:
            raise ClusterError("unreachable", f"{resp.status_code} for {path}")
        return resp.json()

    kubectl = shutil.which("kubectl")
    if kubectl is None:
        raise ClusterError("unavailable")
    try:
        done = subprocess.run(
            [kubectl, "get", "--raw", path], capture_output=True, text=True, timeout=_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClusterError("unreachable", str(exc)) from exc
    if done.returncode != 0:
        kind = "forbidden" if "forbidden" in done.stderr.lower() else "unreachable"
        raise ClusterError(kind, done.stderr.strip())
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise ClusterError("unreachable", "kubectl did not return JSON") from exc


def _node_ready(node: dict) -> bool:
    return any(
        c.get("type") == "Ready" and c.get("status") == "True"
        for c in node.get("status", {}).get("conditions", [])
    )


def _trouble(pod: dict) -> str | None:
    """What is wrong with a pod, as the end of a sentence; ``None`` when it is
    running with every container ready, or finished."""
    status = pod.get("status", {})
    phase = status.get("phase")
    if phase == "Succeeded":
        return None
    containers = status.get("containerStatuses", [])
    if phase == "Running" and containers and all(c.get("ready") for c in containers):
        return None
    reasons = [status.get("reason")]
    for c in status.get("initContainerStatuses", []) + containers:
        if not c.get("ready"):
            state = c.get("state", {})
            reasons.append((state.get("waiting") or state.get("terminated") or {}).get("reason"))
    for reason in reasons:
        if reason in _REASONS:
            return _REASONS[reason]
    if phase == "Pending":
        return "is pending"
    if phase == "Failed":
        return "has failed"
    return "is not ready"


def _spoken_name(pod: dict) -> str:
    """The pod by what owns it — "jarvis-brain in jarvis", not the hash."""
    meta = pod.get("metadata", {})
    name = meta.get("name", "a pod")
    owners = meta.get("ownerReferences") or []
    if owners:
        name = owners[0].get("name") or name
        if owners[0].get("kind") == "ReplicaSet" and "-" in name:
            name = name.rsplit("-", 1)[0]  # the Deployment's name
    return f"{name} in {meta.get('namespace', 'default')}"


def _listed(parts: list[str]) -> str:
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _nodes_line(nodes: list[dict]) -> tuple[str, bool]:
    down = [n.get("metadata", {}).get("name", "a node") for n in nodes if not _node_ready(n)]
    total = len(nodes)
    if not down:
        return ("The node is ready, sir" if total == 1 else f"All {total} nodes are ready, sir"), True
    ready = total - len(down)
    verb = "is" if ready == 1 else "are"
    names = _listed(down)
    return (
        f"{ready} of {total} nodes {verb} ready, sir: {names} "
        f"{'is' if len(down) == 1 else 'are'} not ready."
    ), False


def run(ctx, **_params) -> str:
    try:
        nodes = _fetch("/api/v1/nodes").get("items", [])
        pods = _fetch("/api/v1/pods").get("items", [])
    except ClusterError as exc:
        log.warning("cluster status: %s (%s)", exc.kind, exc)
        return {
            "forbidden": "I am not permitted to read the cluster, sir.",
            "unavailable": "I have no way to reach the cluster from here, sir.",
        }.get(exc.kind, "I can't reach the cluster, sir.")

    live = [(p, _trouble(p)) for p in pods if p.get("status", {}).get("phase") != "Succeeded"]
    bad = [(p, why) for p, why in live if why]
    nodes_line, nodes_ok = _nodes_line(nodes)

    if not bad:
        count = len(live)
        pods_line = (
            "the one pod is running" if count == 1 else f"all {count} pods are running"
        )
        if nodes_ok:
            return f"{nodes_line}, and {pods_line}."
        return f"{nodes_line} {pods_line[:1].upper()}{pods_line[1:]}."

    named = [f"{_spoken_name(p)} {why}" for p, why in bad[:_MAX_NAMED]]
    more = len(bad) - len(named)
    if more:
        named.append(f"{more} more")
    verb = "is" if len(bad) == 1 else "are"
    pods_line = f"{len(bad)} of {len(live)} pods {verb} not healthy: {_listed(named)}."
    return f"{nodes_line}{'.' if nodes_ok else ''} {pods_line}"
