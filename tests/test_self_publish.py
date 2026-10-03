"""M8 part A: every rewrite JARVIS keeps of a builtin is committed to the
``jarvis/self`` branch on GitHub (the owner's choice: its own branch,
pushed; ``develop`` stays theirs). Through GitHub's contents API, because the
pod has no ``git``. Tested against a fake GitHub (``httpx.MockTransport``)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from jarvis.learning.publish import SelfPublisher


class FakeGitHub:
    def __init__(self, branch_exists=False, file_sha=None):
        self.branch_exists = branch_exists
        self.file_sha = file_sha
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        assert request.headers["authorization"] == "Bearer tok"
        if request.method == "GET" and path.endswith("/git/ref/heads/jarvis/self"):
            return httpx.Response(200 if self.branch_exists else 404, json={"object": {"sha": "self1"}})
        if request.method == "GET" and path.endswith("/git/ref/heads/develop"):
            return httpx.Response(200, json={"object": {"sha": "dev1"}})
        if request.method == "POST" and path.endswith("/git/refs"):
            self.branch_exists = True
            return httpx.Response(201, json={})
        if request.method == "GET" and "/contents/" in path:
            if self.file_sha is None:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"sha": self.file_sha})
        if request.method == "PUT" and "/contents/" in path:
            return httpx.Response(200, json={"commit": {"sha": "c0ffee"}})
        return httpx.Response(500)


def publisher(fake, token="tok"):
    return SelfPublisher(token, client=httpx.Client(transport=httpx.MockTransport(fake)))


def test_without_a_token_nothing_is_published():
    p = SelfPublisher("")
    assert not p.available
    assert p.publish("jarvis/skills/builtin/search.py", "x", "msg") is None


def test_the_branch_is_made_from_develop_when_missing_and_the_file_committed():
    fake = FakeGitHub()
    sha = publisher(fake).publish("jarvis/skills/builtin/search.py", "print('hi')\n", "JARVIS rewrote search")
    assert sha == "c0ffee"
    made = next(r for r in fake.requests if r.method == "POST")
    assert json.loads(made.content) == {"ref": "refs/heads/jarvis/self", "sha": "dev1"}
    put = next(r for r in fake.requests if r.method == "PUT")
    body = json.loads(put.content)
    assert body["branch"] == "jarvis/self" and "sha" not in body
    assert base64.b64decode(body["content"]).decode() == "print('hi')\n"
    assert body["author"]["email"] == "nocktok123@gmail.com"
    assert "JARVIS" in body["author"]["name"]


def test_an_existing_file_is_updated_by_its_sha():
    fake = FakeGitHub(branch_exists=True, file_sha="old1")
    publisher(fake).publish("jarvis/skills/builtin/search.py", "x", "msg")
    put = next(r for r in fake.requests if r.method == "PUT")
    assert json.loads(put.content)["sha"] == "old1"
    assert not any(r.method == "POST" for r in fake.requests)


def test_it_never_writes_develop_or_master():
    with pytest.raises(ValueError):
        SelfPublisher("tok", branch="develop")
    with pytest.raises(ValueError):
        SelfPublisher("tok", branch="master")


def test_a_github_error_is_a_none_not_a_crash():
    def broken(request):
        return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    p = SelfPublisher("tok", client=httpx.Client(transport=httpx.MockTransport(broken)))
    assert p.publish("jarvis/skills/builtin/search.py", "x", "msg") is None
    assert "403" in p.last_error


def test_the_token_comes_from_the_environment(monkeypatch, tmp_path):
    from jarvis.config import load_config

    monkeypatch.setenv("JARVIS_GITHUB_TOKEN", "ghp_x")
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    assert load_config(tmp_path / "config.toml").github_token == "ghp_x"
