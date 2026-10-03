"""M8: JARVIS's own commits — each rewrite it keeps of a builtin, pushed to
the ``jarvis/self`` branch of the repo (the owner's choice: its own branch,
pushed; ``develop`` and ``master`` stay theirs to merge).

Through GitHub's REST contents API rather than ``git``: the pod has no git,
and one file per commit is all a rewrite is. The token
(``JARVIS_GITHUB_TOKEN``, a fine-grained token with contents:write on this
repository only) lives in ``.env`` / the pod's ``jarvis-env`` Secret. A
repository ruleset on ``develop`` and ``master`` is what keeps that token off
them; this class refuses them as well.
"""

from __future__ import annotations

import base64
import logging

import httpx

log = logging.getLogger(__name__)

API = "https://api.github.com"
#: who JARVIS's commits are by: the owner's address (CLAUDE.md's authorship
#: rule), with a name that says it was JARVIS
AUTHOR = {"name": "JARVIS (on behalf of Kevin Lundell)", "email": "nocktok123@gmail.com"}
_PROTECTED = frozenset({"develop", "master", "main"})


class SelfPublisher:
    def __init__(
        self,
        token: str | None,
        *,
        repo: str = "Nirvana77/JARVIS",
        branch: str = "jarvis/self",
        base: str = "develop",
        client: httpx.Client | None = None,
    ) -> None:
        if branch in _PROTECTED:
            raise ValueError(f"JARVIS never commits to {branch!r}")
        self.token = token or ""
        self.repo = repo
        self.branch = branch
        self.base = base
        self._client = client
        self.last_error = ""

    @classmethod
    def from_config(cls, config) -> "SelfPublisher":
        s = config.learning
        return cls(config.github_token, repo=s.publish_repo, branch=s.publish_branch, base=s.publish_base)

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=20.0)
        return self._client

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return self._http().request(method, f"{API}/repos/{self.repo}{path}", headers=headers, **kw)

    def publish(self, path: str, content: str, message: str) -> str | None:
        """Commit ``content`` as ``path`` on the branch; the commit's sha, or
        ``None`` (no token, or GitHub said no — ``last_error`` says why).
        Blocking: call it off the event loop."""
        if not self.available:
            return None
        try:
            self._ensure_branch()
            current = self._call("GET", f"/contents/{path}", params={"ref": self.branch})
            body = {
                "message": message,
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                "branch": self.branch,
                "author": AUTHOR,
                "committer": AUTHOR,
            }
            if current.status_code == 200:
                body["sha"] = current.json()["sha"]
            elif current.status_code != 404:
                current.raise_for_status()
            put = self._call("PUT", f"/contents/{path}", json=body)
            put.raise_for_status()
            sha = put.json()["commit"]["sha"]
            log.info("published %s to %s@%s (%s)", path, self.repo, self.branch, sha[:7])
            self.last_error = ""
            return sha
        except Exception as exc:  # noqa: BLE001 — publishing never costs the rewrite
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("could not publish %s to %s: %s", path, self.branch, self.last_error)
            return None

    def _ensure_branch(self) -> None:
        found = self._call("GET", f"/git/ref/heads/{self.branch}")
        if found.status_code == 200:
            return
        if found.status_code != 404:
            found.raise_for_status()
        base = self._call("GET", f"/git/ref/heads/{self.base}")
        base.raise_for_status()
        made = self._call(
            "POST", "/git/refs",
            json={"ref": f"refs/heads/{self.branch}", "sha": base.json()["object"]["sha"]},
        )
        made.raise_for_status()
