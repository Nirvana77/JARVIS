"""Optional local LLM client (Ollama HTTP), capability-probed at startup.

Used only to style-rewrite dynamic lines into the active persona's voice. It is
never on the hot path for understanding or dispatch, and JARVIS is fully
functional without it — ``available`` is ``False`` and callers fall back to
plain phrasing / canned lines.
"""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)


class Reasoner:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.available = False

    @classmethod
    def from_config(cls, config) -> "Reasoner":
        r = cls(
            base_url=config.reasoner.base_url,
            model=config.reasoner.model,
        )
        if config.reasoner.enabled:
            r.probe()
        else:
            log.info("reasoner disabled in config")
        return r

    def probe(self) -> bool:
        """Check a reachable Ollama with the configured model. Never raises."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            # match "qwen2.5:3b" against "qwen2.5:3b" or a bare "qwen2.5"
            self.available = any(
                n == self.model or n.split(":")[0] == self.model.split(":")[0]
                for n in names
            )
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            log.info("reasoner unavailable (%s): %s", self.base_url, exc)
            self.available = False
        log.info(
            "reasoner tier: %s", "ollama:" + self.model if self.available else "none"
        )
        return self.available

    def generate(self, system: str, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
