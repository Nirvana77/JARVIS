"""The `cluster_status` builtin: "check the kubernetes status" answered from
the cluster's own API, with nothing for the speaker to dictate.

Nothing here reaches a cluster: `_fetch` (or what it calls) is replaced."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from jarvis.core.context import Context
from jarvis.skills.builtin import cluster_status


def _ctx(config, tmp_path):
    return Context(say=lambda _t: None, config=config, _data_dir=tmp_path, llm=None)


def node(name, ready=True):
    return {
        "metadata": {"name": name},
        "status": {"conditions": [
            {"type": "MemoryPressure", "status": "False"},
            {"type": "Ready", "status": "True" if ready else "Unknown"},
        ]},
    }


def pod(name, namespace="default", phase="Running", ready=True, waiting=None, owner=None):
    state = {"waiting": {"reason": waiting}} if waiting else {"running": {}}
    meta = {"name": name, "namespace": namespace}
    if owner:
        meta["ownerReferences"] = [{"kind": owner[0], "name": owner[1]}]
    return {
        "metadata": meta,
        "status": {"phase": phase, "containerStatuses": [{"ready": ready, "state": state}]},
    }


def cluster(monkeypatch, nodes, pods):
    asked = []

    def fetch(path):
        asked.append(path)
        return {"items": nodes if "nodes" in path else pods}

    monkeypatch.setattr(cluster_status, "_fetch", fetch)
    return asked


def test_a_healthy_cluster_is_one_short_line(config, tmp_path, monkeypatch):
    asked = cluster(
        monkeypatch,
        [node("kevin-ai"), node("nas"), node("pi")],
        [pod(f"web-{i}") for i in range(4)] + [pod("backup-1", phase="Succeeded", ready=False)],
    )
    line = cluster_status.run(_ctx(config, tmp_path))
    assert line == "All 3 nodes are ready, sir, and all 4 pods are running."
    assert asked == ["/api/v1/nodes", "/api/v1/pods"]


def test_it_takes_no_params_so_there_is_nothing_to_dictate(config, tmp_path, monkeypatch):
    """What this replaces asked for "the kubectl command itself"."""
    cluster(monkeypatch, [node("kevin-ai")], [pod("web")])
    assert cluster_status.MANIFEST.params == {}
    line = cluster_status.run(_ctx(config, tmp_path), command="kubectl get pods")
    assert line == "The node is ready, sir, and the one pod is running."


def test_a_node_that_is_not_ready_is_named(config, tmp_path, monkeypatch):
    cluster(monkeypatch, [node("kevin-ai"), node("nas", ready=False)], [pod("web")])
    line = cluster_status.run(_ctx(config, tmp_path))
    assert line.startswith("1 of 2 nodes is ready, sir: nas is not ready.")


def test_unhealthy_pods_are_named_by_what_owns_them(config, tmp_path, monkeypatch):
    cluster(monkeypatch, [node("kevin-ai")], [
        pod("web"),
        pod("jarvis-brain-5597b99545-ljgnh", "jarvis", ready=False, waiting="CrashLoopBackOff",
            owner=("ReplicaSet", "jarvis-brain-5597b99545")),
        pod("tranga-0", "media", phase="Pending", ready=False, waiting="ImagePullBackOff",
            owner=("StatefulSet", "tranga")),
    ])
    line = cluster_status.run(_ctx(config, tmp_path))
    assert line == (
        "The node is ready, sir. 2 of 3 pods are not healthy: jarvis-brain in jarvis is "
        "crash looping, and tranga in media cannot pull its image."
    )
    assert "5597b99545" not in line  # nobody wants a pod hash read to them


def test_a_long_list_of_unhealthy_pods_is_cut_short(config, tmp_path, monkeypatch):
    bad = [pod(f"job-{i}", "batch", phase="Pending", ready=False) for i in range(6)]
    cluster(monkeypatch, [node("kevin-ai")], bad + [pod("web")])
    line = cluster_status.run(_ctx(config, tmp_path))
    assert "6 of 7 pods are not healthy" in line
    assert line.count("is pending") == 3
    assert line.endswith("and 3 more.")


@pytest.mark.parametrize("kind,said", [
    ("forbidden", "not permitted"),
    ("unreachable", "can't reach the cluster"),
    ("unavailable", "no way to reach the cluster"),
])
def test_a_cluster_it_cannot_read_is_said_plainly(config, tmp_path, monkeypatch, kind, said):
    def fetch(path):
        raise cluster_status.ClusterError(kind)

    monkeypatch.setattr(cluster_status, "_fetch", fetch)
    line = cluster_status.run(_ctx(config, tmp_path))
    assert said in line and "sir" in line


# -- how it reaches the API ------------------------------------------------------------

def test_inside_a_pod_it_uses_the_service_account(tmp_path, monkeypatch):
    (tmp_path / "token").write_text("s3cret\n")
    (tmp_path / "ca.crt").write_text("CA")
    monkeypatch.setattr(cluster_status, "_SA_DIR", tmp_path)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {"items": []})

    monkeypatch.setattr(cluster_status.requests, "get", get)
    assert cluster_status._fetch("/api/v1/nodes") == {"items": []}
    url, kwargs = calls[0]
    assert url == "https://10.43.0.1:443/api/v1/nodes"
    assert kwargs["headers"] == {"Authorization": "Bearer s3cret"}
    assert kwargs["verify"] == str(tmp_path / "ca.crt")
    assert kwargs["timeout"]


def test_a_service_account_without_the_role_is_forbidden(tmp_path, monkeypatch):
    (tmp_path / "token").write_text("s3cret")
    monkeypatch.setattr(cluster_status, "_SA_DIR", tmp_path)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
    monkeypatch.setattr(
        cluster_status.requests, "get", lambda url, **kw: SimpleNamespace(status_code=403)
    )
    with pytest.raises(cluster_status.ClusterError) as err:
        cluster_status._fetch("/api/v1/nodes")
    assert err.value.kind == "forbidden"


def test_outside_a_pod_it_asks_kubectl(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_status, "_SA_DIR", tmp_path / "absent")
    monkeypatch.setattr(cluster_status.shutil, "which", lambda name: "/usr/local/bin/kubectl")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"items": [1]}), stderr="")

    monkeypatch.setattr(cluster_status.subprocess, "run", run)
    assert cluster_status._fetch("/api/v1/pods") == {"items": [1]}
    argv, kwargs = calls[0]
    assert argv == ["/usr/local/bin/kubectl", "get", "--raw", "/api/v1/pods"]
    assert kwargs["timeout"]


def test_with_neither_it_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_status, "_SA_DIR", tmp_path / "absent")
    monkeypatch.setattr(cluster_status.shutil, "which", lambda name: None)
    with pytest.raises(cluster_status.ClusterError) as err:
        cluster_status._fetch("/api/v1/nodes")
    assert err.value.kind == "unavailable"


def test_it_starts_no_thread(config, tmp_path, monkeypatch):
    cluster(monkeypatch, [node("kevin-ai")], [pod("web")])
    before = set(threading.enumerate())
    cluster_status.run(_ctx(config, tmp_path))
    assert set(threading.enumerate()) == before
